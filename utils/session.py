"""세션 관리자 - 다중 사용자 지원을 위한 세션 관리"""

import uuid
import re
import time
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import asyncio

from config.defaults import DATA_DIR, OUTPUTS_DIR


# 세션 데이터 디렉토리
SESSIONS_DIR = DATA_DIR / "sessions"


@dataclass
class SessionInfo:
    """세션 정보"""
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    request_count: int = 0  # Rate limiting용
    request_reset_time: float = field(default_factory=time.time)
    
    def update_activity(self):
        """활동 시간 업데이트"""
        self.last_activity = time.time()
    
    def increment_request(self) -> int:
        """요청 카운트 증가 및 현재 분당 요청 수 반환"""
        current_time = time.time()
        # 1분 경과시 카운트 리셋
        if current_time - self.request_reset_time >= 60:
            self.request_count = 0
            self.request_reset_time = current_time
        
        self.request_count += 1
        return self.request_count
    
    def get_data_dir(self) -> Path:
        """세션별 데이터 디렉토리 경로"""
        path = SESSIONS_DIR / self.session_id
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_outputs_dir(self) -> Path:
        """세션별 출력 디렉토리 경로"""
        path = OUTPUTS_DIR / self.session_id
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "session_id": self.session_id,
            "created_at": datetime.fromtimestamp(self.created_at).isoformat(),
            "last_activity": datetime.fromtimestamp(self.last_activity).isoformat(),
            "data_size": self._get_data_size(),
        }
    
    def _get_data_size(self) -> str:
        """세션 데이터 크기 계산"""
        total_size = 0
        
        # 세션 데이터 디렉토리
        data_dir = SESSIONS_DIR / self.session_id
        if data_dir.exists():
            for f in data_dir.rglob("*"):
                if f.is_file():
                    total_size += f.stat().st_size
        
        # 출력 디렉토리
        outputs_dir = OUTPUTS_DIR / self.session_id
        if outputs_dir.exists():
            for f in outputs_dir.rglob("*"):
                if f.is_file():
                    total_size += f.stat().st_size
        
        # 사람이 읽기 쉬운 형식으로 변환
        if total_size < 1024:
            return f"{total_size} B"
        elif total_size < 1024 * 1024:
            return f"{total_size / 1024:.1f} KB"
        elif total_size < 1024 * 1024 * 1024:
            return f"{total_size / (1024 * 1024):.1f} MB"
        else:
            return f"{total_size / (1024 * 1024 * 1024):.2f} GB"


class SessionManager:
    """세션 관리자"""
    
    COOKIE_NAME = "z_image_session"
    COOKIE_MAX_AGE = 30 * 24 * 60 * 60  # 30일
    RATE_LIMIT_PER_MINUTE = 10  # 분당 최대 요청 수
    
    # UUID 형식 검증용 정규식 (디렉토리 트래버설 방지)
    UUID_PATTERN = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
    )
    
    def __init__(self):
        self._sessions: Dict[str, SessionInfo] = {}
        self._lock = asyncio.Lock()
        
        # 디렉토리 생성
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        
        # 기존 세션 복원
        self._restore_sessions()
    
    def _restore_sessions(self):
        """기존 세션 디렉토리에서 세션 복원"""
        if SESSIONS_DIR.exists():
            for session_dir in SESSIONS_DIR.iterdir():
                if session_dir.is_dir() and self.UUID_PATTERN.match(session_dir.name):
                    session_id = session_dir.name
                    # 마지막 수정 시간을 last_activity로 사용
                    try:
                        mtime = session_dir.stat().st_mtime
                        self._sessions[session_id] = SessionInfo(
                            session_id=session_id,
                            created_at=mtime,
                            last_activity=mtime
                        )
                    except Exception:
                        pass
    
    def validate_session_id(self, session_id: str) -> bool:
        """세션 ID 유효성 검증 (보안: 디렉토리 트래버설 방지)"""
        if not session_id:
            return False
        return bool(self.UUID_PATTERN.match(session_id))
    
    def generate_session_id(self) -> str:
        """새 세션 ID 생성"""
        return str(uuid.uuid4())
    
    async def get_or_create_session(self, session_id: Optional[str] = None) -> SessionInfo:
        """세션 가져오기 또는 생성"""
        async with self._lock:
            # 유효한 세션 ID가 있으면 기존 세션 반환
            if session_id and self.validate_session_id(session_id):
                if session_id in self._sessions:
                    session = self._sessions[session_id]
                    session.update_activity()
                    return session
                else:
                    # 디렉토리가 있으면 세션 복원
                    session_dir = SESSIONS_DIR / session_id
                    if session_dir.exists():
                        session = SessionInfo(session_id=session_id)
                        self._sessions[session_id] = session
                        return session
            
            # 새 세션 생성
            new_id = self.generate_session_id()
            session = SessionInfo(session_id=new_id)
            self._sessions[new_id] = session
            
            # 디렉토리 생성
            session.get_data_dir()
            session.get_outputs_dir()
            
            return session
    
    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """세션 가져오기 (없으면 None)"""
        if not self.validate_session_id(session_id):
            return None
        return self._sessions.get(session_id)
    
    def check_rate_limit(self, session_id: str) -> tuple[bool, int]:
        """
        Rate limit 체크
        Returns: (초과 여부, 현재 요청 수)
        """
        session = self.get_session(session_id)
        if not session:
            return False, 0
        
        count = session.increment_request()
        exceeded = count > self.RATE_LIMIT_PER_MINUTE
        return exceeded, count
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """모든 세션 목록 (관리자용)"""
        sessions = []
        
        # 메모리에 있는 세션
        for session in self._sessions.values():
            sessions.append(session.to_dict())
        
        # 메모리에 없지만 디렉토리가 있는 세션
        if SESSIONS_DIR.exists():
            for session_dir in SESSIONS_DIR.iterdir():
                if session_dir.is_dir() and self.UUID_PATTERN.match(session_dir.name):
                    session_id = session_dir.name
                    if session_id not in self._sessions:
                        try:
                            mtime = session_dir.stat().st_mtime
                            temp_session = SessionInfo(
                                session_id=session_id,
                                created_at=mtime,
                                last_activity=mtime
                            )
                            sessions.append(temp_session.to_dict())
                        except Exception:
                            pass
        
        # 마지막 활동 시간 기준 정렬
        sessions.sort(key=lambda x: x["last_activity"], reverse=True)
        return sessions
    
    async def delete_session(self, session_id: str) -> bool:
        """세션 삭제 (관리자용)"""
        if not self.validate_session_id(session_id):
            return False
        
        async with self._lock:
            # 메모리에서 제거
            if session_id in self._sessions:
                del self._sessions[session_id]
            
            # 세션 데이터 디렉토리 삭제
            session_data_dir = SESSIONS_DIR / session_id
            if session_data_dir.exists():
                try:
                    shutil.rmtree(session_data_dir)
                except Exception as e:
                    print(f"세션 데이터 삭제 실패: {e}")
            
            # 출력 디렉토리 삭제
            outputs_dir = OUTPUTS_DIR / session_id
            if outputs_dir.exists():
                try:
                    shutil.rmtree(outputs_dir)
                except Exception as e:
                    print(f"출력 디렉토리 삭제 실패: {e}")
            
            return True
    
    def get_active_session_count(self) -> int:
        """활성 세션 수 (현재 연결된 WebSocket 수와 별개)"""
        return len(self._sessions)


# 전역 인스턴스
session_manager = SessionManager()


def is_localhost(client_host: Optional[str]) -> bool:
    """localhost 여부 확인 (관리자 권한 체크용)"""
    if not client_host:
        return False
    
    localhost_addresses = {
        "127.0.0.1",
        "::1",
        "localhost",
        "0.0.0.0",
    }
    
    return client_host in localhost_addresses

