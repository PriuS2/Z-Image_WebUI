"""프롬프트 히스토리 관리"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from config.defaults import DATA_DIR


@dataclass
class HistoryEntry:
    """히스토리 항목"""
    id: str
    prompt: str
    negative_prompt: str
    settings: Dict[str, Any]
    timestamp: str
    image_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HistoryEntry":
        return cls(**data)


class HistoryManager:
    """프롬프트 히스토리 관리"""
    
    MAX_HISTORY = 100  # 최대 저장 개수
    
    def __init__(self, history_file: Optional[Path] = None):
        self.history_file = history_file or DATA_DIR / "history.json"
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self._history: List[HistoryEntry] = []
        self._load()
    
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
        """히스토리 파일 저장"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump([entry.to_dict() for entry in self._history], f, 
                         indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"히스토리 저장 실패: {e}")
    
    def add(
        self,
        prompt: str,
        negative_prompt: str = "",
        settings: Optional[Dict[str, Any]] = None,
        image_path: Optional[str] = None
    ) -> HistoryEntry:
        """히스토리 항목 추가"""
        entry = HistoryEntry(
            id=datetime.now().strftime("%Y%m%d%H%M%S%f"),
            prompt=prompt,
            negative_prompt=negative_prompt,
            settings=settings or {},
            timestamp=datetime.now().isoformat(),
            image_path=image_path
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


# 전역 인스턴스
history_manager = HistoryManager()
