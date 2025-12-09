"""즐겨찾기 관리"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from config.defaults import DATA_DIR


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
    
    def __init__(self, favorites_file: Optional[Path] = None):
        self.favorites_file = favorites_file or DATA_DIR / "favorites.json"
        self.favorites_file.parent.mkdir(parents=True, exist_ok=True)
        self._favorites: List[FavoriteEntry] = []
        self._load()
    
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
        """즐겨찾기 파일 저장"""
        try:
            with open(self.favorites_file, 'w', encoding='utf-8') as f:
                json.dump([entry.to_dict() for entry in self._favorites], f, 
                         indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"즐겨찾기 저장 실패: {e}")
    
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


# 전역 인스턴스
favorites_manager = FavoritesManager()
