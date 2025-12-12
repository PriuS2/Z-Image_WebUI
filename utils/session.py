"""세션 관리자 - 계정 기반 다중 사용자 지원을 위한 세션 관리"""

import uuid
import re
import time
import shutil
import json
import socket
import hashlib
import secrets
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
    session_id: str  # 랜덤 UUID (쿠키용)
    user_id: Optional[int] = None  # 로그인한 사용자 ID
    username: Optional[str] = None  # 로그인한 사용자 이름
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    request_count: int = 0  # Rate limiting용
    request_reset_time: float = field(default_factory=time.time)
    
    @property
    def is_authenticated(self) -> bool:
        """로그인 여부"""
        return self.user_id is not None
    
    @property
    def data_id(self) -> str:
        """데이터 저장용 ID (user_{user_id} 또는 session_{session_id})"""
        if self.user_id:
            return f"user_{self.user_id}"
        return f"session_{self.session_id[:8]}"  # 비로그인 시 세션 ID 앞 8자리 사용
    
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
        """세션/사용자별 데이터 디렉토리 경로"""
        path = SESSIONS_DIR / self.data_id
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_outputs_dir(self) -> Path:
        """세션/사용자별 출력 디렉토리 경로"""
        path = OUTPUTS_DIR / self.data_id
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "username": self.username,
            "is_authenticated": self.is_authenticated,
            "data_id": self.data_id,
            "created_at": datetime.fromtimestamp(self.created_at).isoformat(),
            "last_activity": datetime.fromtimestamp(self.last_activity).isoformat(),
            "data_size": self._get_data_size(),
        }
    
    def _get_data_size(self) -> str:
        """세션 데이터 크기 계산"""
        total_size = 0
        
        # 세션 데이터 디렉토리
        data_dir = SESSIONS_DIR / self.data_id
        if data_dir.exists():
            for f in data_dir.rglob("*"):
                if f.is_file():
                    total_size += f.stat().st_size
        
        # 출력 디렉토리
        outputs_dir = OUTPUTS_DIR / self.data_id
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
    
    def get_settings_file(self) -> Path:
        """세션별 설정 파일 경로"""
        return self.get_data_dir() / "settings.json"
    
    def get_settings(self) -> Dict[str, Any]:
        """세션별 설정 로드"""
        settings_file = self.get_settings_file()
        if settings_file.exists():
            try:
                with open(settings_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"세션 설정 로드 실패: {e}")
        return {}
    
    def save_settings(self, settings: Dict[str, Any]) -> bool:
        """세션별 설정 저장"""
        settings_file = self.get_settings_file()
        try:
            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"세션 설정 저장 실패: {e}")
            return False
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """특정 설정값 가져오기"""
        settings = self.get_settings()
        return settings.get(key, default)
    
    def set_setting(self, key: str, value: Any) -> bool:
        """특정 설정값 저장"""
        settings = self.get_settings()
        settings[key] = value
        return self.save_settings(settings)
    
    def login(self, user_id: int, username: str):
        """로그인 처리"""
        self.user_id = user_id
        self.username = username
        self.update_activity()
    
    def logout(self):
        """로그아웃 처리"""
        self.user_id = None
        self.username = None


class SessionManager:
    """세션 관리자"""
    
    COOKIE_NAME = "z_image_session"
    COOKIE_MAX_AGE = 30 * 24 * 60 * 60  # 30일
    RATE_LIMIT_PER_MINUTE = 10  # 분당 최대 요청 수
    
    # 세션 ID 형식 검증용 정규식 (디렉토리 트래버설 방지)
    # UUID 형식
    UUID_PATTERN = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
    )
    # user_숫자 또는 session_문자열 형식
    DATA_ID_PATTERN = re.compile(r'^(user_\d+|session_[a-zA-Z0-9_-]+)$')
    # 기존 컴퓨터 이름 기반 ID도 허용 (호환성)
    LEGACY_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,64}$')
    
    def __init__(self):
        self._sessions: Dict[str, SessionInfo] = {}  # session_id -> SessionInfo
        self._user_sessions: Dict[int, str] = {}  # user_id -> session_id (활성 로그인 매핑)
        self._lock = asyncio.Lock()
        
        # 디렉토리 생성
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    def _generate_session_id(self) -> str:
        """새 세션 ID 생성 (UUID)"""
        return str(uuid.uuid4())
    
    def validate_session_id(self, session_id: str) -> bool:
        """세션 ID 유효성 검증 (보안: 디렉토리 트래버설 방지)"""
        if not session_id:
            return False
        return bool(self.UUID_PATTERN.match(session_id) or self.LEGACY_PATTERN.match(session_id))
    
    def validate_data_id(self, data_id: str) -> bool:
        """데이터 ID 유효성 검증"""
        if not data_id:
            return False
        return bool(self.DATA_ID_PATTERN.match(data_id) or self.LEGACY_PATTERN.match(data_id))
    
    async def get_or_create_session(self, session_id: Optional[str] = None) -> SessionInfo:
        """
        세션 가져오기 또는 생성
        - 쿠키에 유효한 세션 ID가 있으면 해당 세션 반환
        - 없으면 새 세션 생성
        """
        async with self._lock:
            # 유효한 세션 ID가 있고 메모리에 있으면 반환
            if session_id and self.validate_session_id(session_id):
                if session_id in self._sessions:
                    session = self._sessions[session_id]
                    session.update_activity()
                    return session
            
            # 새 세션 생성
            new_session_id = self._generate_session_id()
            session = SessionInfo(session_id=new_session_id)
            self._sessions[new_session_id] = session
            
            return session
    
    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """세션 가져오기 (없으면 None)"""
        if not self.validate_session_id(session_id):
            return None
        return self._sessions.get(session_id)
    
    def get_session_by_user(self, user_id: int) -> Optional[SessionInfo]:
        """사용자 ID로 활성 세션 가져오기"""
        session_id = self._user_sessions.get(user_id)
        if session_id:
            return self._sessions.get(session_id)
        return None
    
    async def login_session(self, session_id: str, user_id: int, username: str) -> bool:
        """세션에 로그인 정보 연결"""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False
            
            # 기존 사용자의 다른 세션이 있으면 로그아웃 처리
            if user_id in self._user_sessions:
                old_session_id = self._user_sessions[user_id]
                if old_session_id in self._sessions:
                    self._sessions[old_session_id].logout()
            
            # 현재 세션에 로그인
            session.login(user_id, username)
            self._user_sessions[user_id] = session_id
            
            # 사용자 데이터 디렉토리 생성
            session.get_data_dir()
            session.get_outputs_dir()
            
            return True
    
    async def logout_session(self, session_id: str) -> bool:
        """세션 로그아웃"""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session or not session.is_authenticated:
                return False
            
            user_id = session.user_id
            session.logout()
            
            # 사용자-세션 매핑 제거
            if user_id and user_id in self._user_sessions:
                del self._user_sessions[user_id]
            
            return True
    
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
        
        # 마지막 활동 시간 기준 정렬
        sessions.sort(key=lambda x: x["last_activity"], reverse=True)
        return sessions
    
    async def delete_session(self, session_id: str) -> bool:
        """세션 삭제 (관리자용)"""
        if not self.validate_session_id(session_id):
            return False
        
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                # 로그인된 세션이면 매핑 제거
                if session.user_id and session.user_id in self._user_sessions:
                    del self._user_sessions[session.user_id]
                
                # 메모리에서 제거
                del self._sessions[session_id]
            
            return True
    
    async def delete_user_data(self, user_id: int) -> bool:
        """사용자 데이터 삭제 (계정 삭제 시)"""
        data_id = f"user_{user_id}"
        
        # 세션 데이터 디렉토리 삭제
        session_data_dir = SESSIONS_DIR / data_id
        if session_data_dir.exists():
            try:
                shutil.rmtree(session_data_dir)
            except Exception as e:
                print(f"세션 데이터 삭제 실패: {e}")
        
        # 출력 디렉토리 삭제
        outputs_dir = OUTPUTS_DIR / data_id
        if outputs_dir.exists():
            try:
                shutil.rmtree(outputs_dir)
            except Exception as e:
                print(f"출력 디렉토리 삭제 실패: {e}")
        
        return True
    
    def get_active_session_count(self) -> int:
        """활성 세션 수"""
        return len(self._sessions)
    
    def get_authenticated_session_count(self) -> int:
        """로그인된 세션 수"""
        return sum(1 for s in self._sessions.values() if s.is_authenticated)


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
