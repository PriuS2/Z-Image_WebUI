"""이미지 생성 큐 관리자 - 다중 사용자 순차 처리"""

import asyncio
import time
from typing import Optional, Dict, Any, List, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid


class QueueItemStatus(Enum):
    """큐 항목 상태"""
    PENDING = "pending"      # 대기 중
    PROCESSING = "processing"  # 처리 중
    COMPLETED = "completed"  # 완료
    FAILED = "failed"        # 실패
    CANCELLED = "cancelled"  # 취소됨


@dataclass
class QueueItem:
    """큐 항목"""
    id: str
    session_id: str
    request_data: Dict[str, Any]
    status: QueueItemStatus = QueueItemStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "status": self.status.value,
            "created_at": datetime.fromtimestamp(self.created_at).isoformat(),
            "started_at": datetime.fromtimestamp(self.started_at).isoformat() if self.started_at else None,
            "completed_at": datetime.fromtimestamp(self.completed_at).isoformat() if self.completed_at else None,
        }


class GenerationQueueManager:
    """이미지 생성 큐 관리자"""
    
    def __init__(self):
        self._queue: List[QueueItem] = []
        self._current_item: Optional[QueueItem] = None
        self._lock = asyncio.Lock()
        self._processing = False
        self._worker_task: Optional[asyncio.Task] = None
        
        # 콜백 함수들
        self._on_status_change: Optional[Callable[[str, str, Dict], Awaitable[None]]] = None
        self._on_broadcast: Optional[Callable[[Dict], Awaitable[None]]] = None
        self._generate_func: Optional[Callable[[Dict], Awaitable[Dict]]] = None
    
    def set_callbacks(
        self,
        on_status_change: Callable[[str, str, Dict], Awaitable[None]],
        on_broadcast: Callable[[Dict], Awaitable[None]],
        generate_func: Callable[[Dict], Awaitable[Dict]]
    ):
        """콜백 함수 설정
        
        Args:
            on_status_change: 세션에 상태 변경 알림 (session_id, event_type, data)
            on_broadcast: 모든 사용자에게 브로드캐스트 (data)
            generate_func: 실제 이미지 생성 함수 (request_data) -> result
        """
        self._on_status_change = on_status_change
        self._on_broadcast = on_broadcast
        self._generate_func = generate_func
    
    async def add_to_queue(
        self,
        session_id: str,
        request_data: Dict[str, Any]
    ) -> tuple[str, int]:
        """
        큐에 요청 추가
        
        Returns:
            (item_id, queue_position)
        """
        async with self._lock:
            item_id = str(uuid.uuid4())
            item = QueueItem(
                id=item_id,
                session_id=session_id,
                request_data=request_data
            )
            self._queue.append(item)
            position = len(self._queue)
            
            # 처리 중인 항목이 있으면 +1
            if self._current_item:
                position += 1
            
            return item_id, position
    
    async def get_queue_position(self, session_id: str, item_id: Optional[str] = None) -> int:
        """
        세션의 큐 위치 반환
        0 = 현재 처리 중, 1 = 다음, -1 = 큐에 없음
        """
        async with self._lock:
            # 현재 처리 중인지 확인
            if self._current_item:
                if item_id and self._current_item.id == item_id:
                    return 0
                if self._current_item.session_id == session_id:
                    return 0
            
            # 큐에서 위치 찾기
            for i, item in enumerate(self._queue):
                if item_id and item.id == item_id:
                    return i + 1
                if not item_id and item.session_id == session_id:
                    return i + 1
            
            return -1
    
    async def get_session_pending_count(self, session_id: str) -> int:
        """세션의 대기 중인 요청 수"""
        async with self._lock:
            count = sum(1 for item in self._queue if item.session_id == session_id)
            if self._current_item and self._current_item.session_id == session_id:
                count += 1
            return count
    
    async def remove_session_items(self, session_id: str) -> int:
        """세션의 모든 대기 중인 항목 제거 (연결 해제 시)"""
        async with self._lock:
            removed = 0
            # 대기 큐에서 제거
            original_len = len(self._queue)
            self._queue = [item for item in self._queue if item.session_id != session_id]
            removed = original_len - len(self._queue)
            
            return removed
    
    async def cancel_item(self, item_id: str, session_id: str) -> bool:
        """특정 항목 취소 (본인 것만)"""
        async with self._lock:
            for i, item in enumerate(self._queue):
                if item.id == item_id and item.session_id == session_id:
                    item.status = QueueItemStatus.CANCELLED
                    self._queue.pop(i)
                    return True
            return False
    
    def get_queue_status(self) -> Dict[str, Any]:
        """전체 큐 상태"""
        return {
            "queue_length": len(self._queue),
            "is_processing": self._current_item is not None,
            "current_session": self._current_item.session_id if self._current_item else None,
        }
    
    def is_processing(self) -> bool:
        """현재 처리 중인지 확인"""
        return self._current_item is not None
    
    def get_current_session(self) -> Optional[str]:
        """현재 처리 중인 세션 ID"""
        return self._current_item.session_id if self._current_item else None
    
    async def start_worker(self):
        """큐 워커 시작"""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._process_queue())
    
    async def stop_worker(self):
        """큐 워커 중지"""
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
    
    async def _process_queue(self):
        """큐 처리 워커"""
        while True:
            try:
                # 다음 항목 가져오기
                item = await self._get_next_item()
                
                if item is None:
                    # 큐가 비어있으면 잠시 대기
                    await asyncio.sleep(0.1)
                    continue
                
                # 처리 시작
                await self._process_item(item)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"큐 워커 에러: {e}")
                await asyncio.sleep(1)
    
    async def _get_next_item(self) -> Optional[QueueItem]:
        """다음 처리할 항목 가져오기"""
        async with self._lock:
            if self._current_item is not None:
                return None
            
            if not self._queue:
                return None
            
            item = self._queue.pop(0)
            item.status = QueueItemStatus.PROCESSING
            item.started_at = time.time()
            self._current_item = item
            
            return item
    
    async def _process_item(self, item: QueueItem):
        """항목 처리"""
        try:
            # 처리 시작 알림
            if self._on_status_change:
                await self._on_status_change(
                    item.session_id,
                    "generation_start",
                    {"item_id": item.id, "position": 0}
                )
            
            # 전체 브로드캐스트: 큐 상태 업데이트
            if self._on_broadcast:
                await self._on_broadcast({
                    "type": "queue_update",
                    "queue_length": len(self._queue),
                    "is_processing": True,
                })
            
            # 대기 중인 다른 사용자들에게 위치 업데이트
            await self._notify_position_updates()
            
            # 실제 생성 실행
            if self._generate_func:
                result = await self._generate_func(item.request_data)
                item.result = result
                item.status = QueueItemStatus.COMPLETED
            else:
                item.status = QueueItemStatus.FAILED
                item.error = "생성 함수가 설정되지 않았습니다."
            
        except Exception as e:
            item.status = QueueItemStatus.FAILED
            item.error = str(e)
            
            # 에러 알림 (해당 사용자에게만)
            if self._on_status_change:
                await self._on_status_change(
                    item.session_id,
                    "generation_error",
                    {"item_id": item.id, "error": str(e)}
                )
        
        finally:
            item.completed_at = time.time()
            
            async with self._lock:
                self._current_item = None
            
            # 완료 알림
            if item.status == QueueItemStatus.COMPLETED and self._on_status_change:
                await self._on_status_change(
                    item.session_id,
                    "generation_complete",
                    {"item_id": item.id, "result": item.result}
                )
            
            # 큐 상태 브로드캐스트
            if self._on_broadcast:
                await self._on_broadcast({
                    "type": "queue_update",
                    "queue_length": len(self._queue),
                    "is_processing": False,
                })
            
            # 대기 중인 사용자들에게 위치 업데이트
            await self._notify_position_updates()
    
    async def _notify_position_updates(self):
        """대기 중인 사용자들에게 위치 업데이트 알림"""
        if not self._on_status_change:
            return
        
        async with self._lock:
            for i, item in enumerate(self._queue):
                position = i + 1
                if self._current_item:
                    position += 1
                
                await self._on_status_change(
                    item.session_id,
                    "queue_position",
                    {"item_id": item.id, "position": position}
                )


# 전역 인스턴스
generation_queue = GenerationQueueManager()

