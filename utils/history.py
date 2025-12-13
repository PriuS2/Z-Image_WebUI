"""프롬프트 히스토리 관리 - 세션별 개인화 지원"""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from config.defaults import DATA_DIR


# 세션별 데이터 디렉토리
SESSIONS_DIR = DATA_DIR / "sessions"


@dataclass
class HistoryEntry:
    """히스토리 항목"""
    id: str
    prompt: str
    settings: Dict[str, Any]
    timestamp: str
    image_path: Optional[str] = None
    conversation: Optional[List[Dict[str, Any]]] = None  # 대화 내용 저장
    korean_prompt: Optional[str] = None  # 한국어 프롬프트 저장
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HistoryEntry":
        # 기존 데이터 호환성 유지
        if "conversation" not in data:
            data["conversation"] = None
        if "korean_prompt" not in data:
            data["korean_prompt"] = None
        return cls(**data)


class HistoryManager:
    """프롬프트 히스토리 관리"""
    
    MAX_HISTORY = 100  # 최대 저장 개수
    
    # 파일 동시 접근 방지용 잠금
    _locks: Dict[str, asyncio.Lock] = {}
    _locks_lock = asyncio.Lock()
    
    def __init__(self, history_file: Optional[Path] = None, session_id: Optional[str] = None):
        """
        Args:
            history_file: 직접 파일 경로 지정 (레거시 호환)
            session_id: 세션 ID (세션별 개인화)
        """
        if session_id:
            # 세션별 히스토리 파일
            session_dir = SESSIONS_DIR / session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            self.history_file = session_dir / "history.json"
        elif history_file:
            self.history_file = history_file
        else:
            # 레거시: 전역 히스토리
            self.history_file = DATA_DIR / "history.json"
        
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self._history: List[HistoryEntry] = []
        self._session_id = session_id
        self._load()
    
    @classmethod
    async def _get_lock(cls, file_path: str) -> asyncio.Lock:
        """파일별 잠금 객체 가져오기"""
        async with cls._locks_lock:
            if file_path not in cls._locks:
                cls._locks[file_path] = asyncio.Lock()
            return cls._locks[file_path]
    
    def _load(self) -> None:
        """히스토리 파일 로드"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._history = [HistoryEntry.from_dict(item) for item in data]
            except Exception as e:
                print(f"히스토리 로드 실패: {e}")
                self._history = []
    
    def _save(self) -> None:
        """히스토리 파일 저장 (동기)"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump([entry.to_dict() for entry in self._history], f, 
                         indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"히스토리 저장 실패: {e}")
    
    async def _save_async(self) -> None:
        """히스토리 파일 저장 (비동기, 잠금 사용)"""
        lock = await self._get_lock(str(self.history_file))
        async with lock:
            try:
                # 파일 I/O는 스레드에서 실행
                await asyncio.to_thread(self._save)
            except Exception as e:
                print(f"히스토리 비동기 저장 실패: {e}")
    
    def add(
        self,
        prompt: str,
        settings: Optional[Dict[str, Any]] = None,
        image_path: Optional[str] = None,
        conversation: Optional[List[Dict[str, Any]]] = None,
        korean_prompt: Optional[str] = None
    ) -> HistoryEntry:
        """히스토리 항목 추가"""
        entry = HistoryEntry(
            id=datetime.now().strftime("%Y%m%d%H%M%S%f"),
            prompt=prompt,
            settings=settings or {},
            timestamp=datetime.now().isoformat(),
            image_path=image_path,
            conversation=conversation,
            korean_prompt=korean_prompt
        )
        
        # 중복 체크 (같은 프롬프트가 있으면 기존 것 제거)
        self._history = [h for h in self._history if h.prompt != prompt]
        
        # 맨 앞에 추가
        self._history.insert(0, entry)
        
        # 최대 개수 제한
        if len(self._history) > self.MAX_HISTORY:
            self._history = self._history[:self.MAX_HISTORY]
        
        self._save()
        return entry
    
    async def add_async(
        self,
        prompt: str,
        settings: Optional[Dict[str, Any]] = None,
        image_path: Optional[str] = None,
        conversation: Optional[List[Dict[str, Any]]] = None,
        korean_prompt: Optional[str] = None
    ) -> HistoryEntry:
        """히스토리 항목 추가 (비동기)"""
        entry = HistoryEntry(
            id=datetime.now().strftime("%Y%m%d%H%M%S%f"),
            prompt=prompt,
            settings=settings or {},
            timestamp=datetime.now().isoformat(),
            image_path=image_path,
            conversation=conversation,
            korean_prompt=korean_prompt
        )
        
        # 중복 체크 (같은 프롬프트가 있으면 기존 것 제거)
        self._history = [h for h in self._history if h.prompt != prompt]
        
        # 맨 앞에 추가
        self._history.insert(0, entry)
        
        # 최대 개수 제한
        if len(self._history) > self.MAX_HISTORY:
            self._history = self._history[:self.MAX_HISTORY]
        
        await self._save_async()
        return entry
    
    def get_all(self) -> List[HistoryEntry]:
        """모든 히스토리 가져오기"""
        return self._history.copy()
    
    def get_recent(self, count: int = 10) -> List[HistoryEntry]:
        """최근 히스토리 가져오기"""
        return self._history[:count]
    
    def get_by_id(self, entry_id: str) -> Optional[HistoryEntry]:
        """ID로 히스토리 가져오기"""
        for entry in self._history:
            if entry.id == entry_id:
                return entry
        return None
    
    def delete(self, entry_id: str) -> bool:
        """히스토리 항목 삭제"""
        original_len = len(self._history)
        self._history = [h for h in self._history if h.id != entry_id]
        if len(self._history) < original_len:
            self._save()
            return True
        return False
    
    def clear(self) -> None:
        """모든 히스토리 삭제"""
        self._history = []
        self._save()
    
    def search(self, query: str) -> List[HistoryEntry]:
        """프롬프트 검색"""
        query_lower = query.lower()
        return [h for h in self._history if query_lower in h.prompt.lower()]
    
    def get_prompts_for_dropdown(self) -> List[str]:
        """드롭다운용 프롬프트 목록 (최근 20개)"""
        return [h.prompt[:80] + "..." if len(h.prompt) > 80 else h.prompt 
                for h in self._history[:20]]


# 세션별 히스토리 매니저 캐시
_session_history_managers: Dict[str, HistoryManager] = {}
_session_managers_lock = asyncio.Lock()


async def get_history_manager(session_id: str) -> HistoryManager:
    """세션별 히스토리 매니저 가져오기 (캐시됨)"""
    async with _session_managers_lock:
        if session_id not in _session_history_managers:
            _session_history_managers[session_id] = HistoryManager(session_id=session_id)
        return _session_history_managers[session_id]


def get_history_manager_sync(session_id: str) -> HistoryManager:
    """세션별 히스토리 매니저 가져오기 (동기, 캐시됨)"""
    if session_id not in _session_history_managers:
        _session_history_managers[session_id] = HistoryManager(session_id=session_id)
    return _session_history_managers[session_id]


def clear_history_manager_cache(session_id: str) -> None:
    """세션별 히스토리 매니저 캐시 제거 (데이터 삭제/초기화용)"""
    _session_history_managers.pop(session_id, None)


# 레거시 호환: 전역 인스턴스 (마이그레이션용)
history_manager = HistoryManager()
