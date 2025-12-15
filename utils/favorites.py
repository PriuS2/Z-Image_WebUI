"""즐겨찾기 관리 - 세션별 개인화 지원"""

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
class FavoriteEntry:
    """즐겨찾기 항목"""
    id: str
    name: str
    prompt: str
    settings: Dict[str, Any]
    created_at: str
    tags: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FavoriteEntry":
        return cls(**data)


class FavoritesManager:
    """프롬프트/설정 즐겨찾기 관리"""
    
    # 파일 동시 접근 방지용 잠금
    _locks: Dict[str, asyncio.Lock] = {}
    _locks_lock = asyncio.Lock()
    
    def __init__(self, favorites_file: Optional[Path] = None, session_id: Optional[str] = None):
        """
        Args:
            favorites_file: 직접 파일 경로 지정 (레거시 호환)
            session_id: 세션 ID (세션별 개인화)
        """
        if session_id:
            # 세션별 즐겨찾기 파일
            session_dir = SESSIONS_DIR / session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            self.favorites_file = session_dir / "favorites.json"
        elif favorites_file:
            self.favorites_file = favorites_file
        else:
            # 레거시: 전역 즐겨찾기
            self.favorites_file = DATA_DIR / "favorites.json"
        
        self.favorites_file.parent.mkdir(parents=True, exist_ok=True)
        self._favorites: List[FavoriteEntry] = []
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
        """즐겨찾기 파일 로드"""
        if self.favorites_file.exists():
            try:
                with open(self.favorites_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._favorites = [FavoriteEntry.from_dict(item) for item in data]
            except Exception as e:
                print(f"즐겨찾기 로드 실패: {e}")
                self._favorites = []
    
    def _save(self) -> None:
        """즐겨찾기 파일 저장 (동기)"""
        try:
            with open(self.favorites_file, 'w', encoding='utf-8') as f:
                json.dump([entry.to_dict() for entry in self._favorites], f, 
                         indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"즐겨찾기 저장 실패: {e}")
    
    async def _save_async(self) -> None:
        """즐겨찾기 파일 저장 (비동기, 잠금 사용)"""
        lock = await self._get_lock(str(self.favorites_file))
        async with lock:
            try:
                await asyncio.to_thread(self._save)
            except Exception as e:
                print(f"즐겨찾기 비동기 저장 실패: {e}")
    
    def add(
        self,
        name: str,
        prompt: str,
        settings: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> FavoriteEntry:
        """즐겨찾기 추가"""
        entry = FavoriteEntry(
            id=datetime.now().strftime("%Y%m%d%H%M%S%f"),
            name=name,
            prompt=prompt,
            settings=settings or {},
            created_at=datetime.now().isoformat(),
            tags=tags or []
        )
        
        self._favorites.append(entry)
        self._save()
        return entry
    
    async def add_async(
        self,
        name: str,
        prompt: str,
        settings: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> FavoriteEntry:
        """즐겨찾기 추가 (비동기)"""
        entry = FavoriteEntry(
            id=datetime.now().strftime("%Y%m%d%H%M%S%f"),
            name=name,
            prompt=prompt,
            settings=settings or {},
            created_at=datetime.now().isoformat(),
            tags=tags or []
        )
        
        self._favorites.append(entry)
        await self._save_async()
        return entry
    
    def get_all(self) -> List[FavoriteEntry]:
        """모든 즐겨찾기 가져오기"""
        return self._favorites.copy()
    
    def get_by_id(self, entry_id: str) -> Optional[FavoriteEntry]:
        """ID로 즐겨찾기 가져오기"""
        for entry in self._favorites:
            if entry.id == entry_id:
                return entry
        return None
    
    def get_by_name(self, name: str) -> Optional[FavoriteEntry]:
        """이름으로 즐겨찾기 가져오기"""
        for entry in self._favorites:
            if entry.name == name:
                return entry
        return None
    
    def update(
        self,
        entry_id: str,
        name: Optional[str] = None,
        prompt: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """즐겨찾기 업데이트"""
        for entry in self._favorites:
            if entry.id == entry_id:
                if name is not None:
                    entry.name = name
                if prompt is not None:
                    entry.prompt = prompt
                if settings is not None:
                    entry.settings = settings
                if tags is not None:
                    entry.tags = tags
                self._save()
                return True
        return False
    
    def delete(self, entry_id: str) -> bool:
        """즐겨찾기 삭제"""
        original_len = len(self._favorites)
        self._favorites = [f for f in self._favorites if f.id != entry_id]
        if len(self._favorites) < original_len:
            self._save()
            return True
        return False
    
    def search_by_tag(self, tag: str) -> List[FavoriteEntry]:
        """태그로 검색"""
        return [f for f in self._favorites if tag in f.tags]
    
    def search_by_name(self, query: str) -> List[FavoriteEntry]:
        """이름으로 검색"""
        query_lower = query.lower()
        return [f for f in self._favorites if query_lower in f.name.lower()]
    
    def get_all_tags(self) -> List[str]:
        """모든 태그 목록"""
        tags = set()
        for entry in self._favorites:
            tags.update(entry.tags)
        return sorted(list(tags))
    
    def get_names_for_dropdown(self) -> List[str]:
        """드롭다운용 이름 목록"""
        return [f.name for f in self._favorites]


# 세션별 즐겨찾기 매니저 캐시
_session_favorites_managers: Dict[str, FavoritesManager] = {}
_session_managers_lock = asyncio.Lock()


async def get_favorites_manager(session_id: str) -> FavoritesManager:
    """세션별 즐겨찾기 매니저 가져오기 (캐시됨)"""
    async with _session_managers_lock:
        if session_id not in _session_favorites_managers:
            _session_favorites_managers[session_id] = FavoritesManager(session_id=session_id)
        return _session_favorites_managers[session_id]


def get_favorites_manager_sync(session_id: str) -> FavoritesManager:
    """세션별 즐겨찾기 매니저 가져오기 (동기, 캐시됨)"""
    if session_id not in _session_favorites_managers:
        _session_favorites_managers[session_id] = FavoritesManager(session_id=session_id)
    return _session_favorites_managers[session_id]


def clear_favorites_manager_cache(session_id: str) -> None:
    """세션별 즐겨찾기 매니저 캐시 제거 (데이터 삭제/초기화용)"""
    _session_favorites_managers.pop(session_id, None)


# 레거시 호환: 전역 인스턴스 (마이그레이션용)
favorites_manager = FavoritesManager()
