"""편집 히스토리 관리 - 세션별 개인화 지원"""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict, field

from config.defaults import DATA_DIR


# 세션별 데이터 디렉토리
SESSIONS_DIR = DATA_DIR / "sessions"


@dataclass
class EditHistoryEntry:
    """편집 히스토리 항목"""
    id: str
    prompt: str  # 편집 프롬프트
    negative_prompt: Optional[str] = None  # 네거티브 프롬프트 (Qwen)
    korean_prompt: Optional[str] = None  # 한국어 프롬프트
    settings: Dict[str, Any] = field(default_factory=dict)  # 편집 설정
    timestamp: str = ""
    original_image_paths: List[str] = field(default_factory=list)  # 원본 이미지 경로들 (1~3장, Qwen)
    result_image_paths: List[str] = field(default_factory=list)  # 결과 이미지 경로들
    parent_id: Optional[str] = None  # 멀티턴 편집 시 이전 편집 ID
    conversation: Optional[List[Dict[str, Any]]] = None  # 대화 내용
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EditHistoryEntry":
        # 기존 데이터 호환성 유지
        defaults = {
            "negative_prompt": None,
            "korean_prompt": None,
            "settings": {},
            "original_image_paths": [],
            "result_image_paths": [],
            "parent_id": None,
            "conversation": None,
        }
        # 기존 original_image_path -> original_image_paths 마이그레이션
        if "original_image_path" in data and "original_image_paths" not in data:
            old_path = data.pop("original_image_path")
            data["original_image_paths"] = [old_path] if old_path else []
        # reference_image_path 제거 (Qwen에서는 사용 안 함)
        data.pop("reference_image_path", None)
        
        for key, default_value in defaults.items():
            if key not in data:
                data[key] = default_value
        return cls(**data)


class EditHistoryManager:
    """편집 히스토리 관리"""
    
    MAX_HISTORY = 100  # 최대 저장 개수
    
    # 파일 동시 접근 방지용 잠금
    _locks: Dict[str, asyncio.Lock] = {}
    _locks_lock = asyncio.Lock()
    
    def __init__(self, history_file: Optional[Path] = None, session_id: Optional[str] = None):
        """
        Args:
            history_file: 직접 파일 경로 지정
            session_id: 세션 ID (세션별 개인화)
        """
        if session_id:
            # 세션별 편집 히스토리 파일
            session_dir = SESSIONS_DIR / session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            self.history_file = session_dir / "edit_history.json"
        elif history_file:
            self.history_file = history_file
        else:
            # 전역 편집 히스토리
            self.history_file = DATA_DIR / "edit_history.json"
        
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self._history: List[EditHistoryEntry] = []
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
                    self._history = [EditHistoryEntry.from_dict(item) for item in data]
            except Exception as e:
                print(f"편집 히스토리 로드 실패: {e}")
                self._history = []
    
    def _save(self) -> None:
        """히스토리 파일 저장 (동기)"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump([entry.to_dict() for entry in self._history], f,
                         indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"편집 히스토리 저장 실패: {e}")
    
    async def _save_async(self) -> None:
        """히스토리 파일 저장 (비동기, 잠금 사용)"""
        lock = await self._get_lock(str(self.history_file))
        async with lock:
            try:
                await asyncio.to_thread(self._save)
            except Exception as e:
                print(f"편집 히스토리 비동기 저장 실패: {e}")
    
    def add(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        original_image_paths: Optional[List[str]] = None,
        result_image_paths: Optional[List[str]] = None,
        parent_id: Optional[str] = None,
        conversation: Optional[List[Dict[str, Any]]] = None,
        korean_prompt: Optional[str] = None
    ) -> EditHistoryEntry:
        """편집 히스토리 항목 추가"""
        entry = EditHistoryEntry(
            id=datetime.now().strftime("%Y%m%d%H%M%S%f"),
            prompt=prompt,
            negative_prompt=negative_prompt,
            korean_prompt=korean_prompt,
            settings=settings or {},
            timestamp=datetime.now().isoformat(),
            original_image_paths=original_image_paths or [],
            result_image_paths=result_image_paths or [],
            parent_id=parent_id,
            conversation=conversation
        )
        
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
        negative_prompt: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        original_image_paths: Optional[List[str]] = None,
        result_image_paths: Optional[List[str]] = None,
        parent_id: Optional[str] = None,
        conversation: Optional[List[Dict[str, Any]]] = None,
        korean_prompt: Optional[str] = None
    ) -> EditHistoryEntry:
        """편집 히스토리 항목 추가 (비동기)"""
        entry = EditHistoryEntry(
            id=datetime.now().strftime("%Y%m%d%H%M%S%f"),
            prompt=prompt,
            negative_prompt=negative_prompt,
            korean_prompt=korean_prompt,
            settings=settings or {},
            timestamp=datetime.now().isoformat(),
            original_image_paths=original_image_paths or [],
            result_image_paths=result_image_paths or [],
            parent_id=parent_id,
            conversation=conversation
        )
        
        self._history.insert(0, entry)
        
        if len(self._history) > self.MAX_HISTORY:
            self._history = self._history[:self.MAX_HISTORY]
        
        await self._save_async()
        return entry
    
    def get_all(self) -> List[EditHistoryEntry]:
        """모든 히스토리 가져오기"""
        return self._history.copy()
    
    def get_recent(self, count: int = 10) -> List[EditHistoryEntry]:
        """최근 히스토리 가져오기"""
        return self._history[:count]
    
    def get_by_id(self, entry_id: str) -> Optional[EditHistoryEntry]:
        """ID로 히스토리 가져오기"""
        for entry in self._history:
            if entry.id == entry_id:
                return entry
        return None
    
    def get_chain(self, entry_id: str) -> List[EditHistoryEntry]:
        """멀티턴 편집 체인 가져오기 (가장 오래된 것부터)"""
        chain = []
        current = self.get_by_id(entry_id)
        
        while current:
            chain.insert(0, current)
            if current.parent_id:
                current = self.get_by_id(current.parent_id)
            else:
                break
        
        return chain
    
    def update_conversation(self, entry_id: str, conversation: List[Dict[str, Any]]) -> bool:
        """대화 내용 업데이트"""
        entry = self.get_by_id(entry_id)
        if entry:
            entry.conversation = conversation
            self._save()
            return True
        return False
    
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
    
    def search(self, query: str) -> List[EditHistoryEntry]:
        """프롬프트 검색"""
        query_lower = query.lower()
        return [h for h in self._history if query_lower in h.prompt.lower()]


# 세션별 편집 히스토리 매니저 캐시
_session_edit_history_managers: Dict[str, EditHistoryManager] = {}
_session_managers_lock = asyncio.Lock()


async def get_edit_history_manager(session_id: str) -> EditHistoryManager:
    """세션별 편집 히스토리 매니저 가져오기 (캐시됨)"""
    async with _session_managers_lock:
        if session_id not in _session_edit_history_managers:
            _session_edit_history_managers[session_id] = EditHistoryManager(session_id=session_id)
        return _session_edit_history_managers[session_id]


def get_edit_history_manager_sync(session_id: str) -> EditHistoryManager:
    """세션별 편집 히스토리 매니저 가져오기 (동기, 캐시됨)"""
    if session_id not in _session_edit_history_managers:
        _session_edit_history_managers[session_id] = EditHistoryManager(session_id=session_id)
    return _session_edit_history_managers[session_id]


def clear_edit_history_manager_cache(session_id: str) -> None:
    """세션별 편집 히스토리 매니저 캐시 제거 (데이터 삭제/초기화용)"""
    _session_edit_history_managers.pop(session_id, None)


# 전역 인스턴스 (마이그레이션용)
edit_history_manager = EditHistoryManager()


