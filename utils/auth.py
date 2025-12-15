"""인증 관리자 - 회원가입, 로그인, 비밀번호 해시/변경"""

import hashlib
import secrets
import string
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from utils.database import get_db


@dataclass
class User:
    """사용자 정보"""
    id: int
    username: str
    created_at: datetime
    last_login: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }


class AuthManager:
    """인증 관리자"""
    
    # 비밀번호 정책
    MIN_PASSWORD_LENGTH = 4
    MAX_PASSWORD_LENGTH = 128
    
    # 아이디 정책
    MIN_USERNAME_LENGTH = 3
    MAX_USERNAME_LENGTH = 50
    
    # 게스트 계정
    GUEST_USERNAME = "guest"
    GUEST_PASSWORD = "guest"
    
    def __init__(self):
        pass
    
    def _hash_password(self, password: str, salt: Optional[str] = None) -> tuple[str, str]:
        """
        비밀번호 해시 (SHA-256 + salt)
        Returns: (hash, salt)
        """
        if salt is None:
            salt = secrets.token_hex(32)
        
        # salt + password를 SHA-256으로 해시
        hash_input = (salt + password).encode('utf-8')
        password_hash = hashlib.sha256(hash_input).hexdigest()
        
        # salt:hash 형식으로 저장
        return f"{salt}:{password_hash}", salt
    
    def _verify_password(self, password: str, stored_hash: str) -> bool:
        """비밀번호 검증"""
        try:
            salt, hash_value = stored_hash.split(':', 1)
            new_hash, _ = self._hash_password(password, salt)
            return new_hash == stored_hash
        except ValueError:
            return False
    
    def _generate_temp_password(self, length: int = 12) -> str:
        """임시 비밀번호 생성"""
        characters = string.ascii_letters + string.digits
        return ''.join(secrets.choice(characters) for _ in range(length))
    
    def validate_username(self, username: str) -> tuple[bool, str]:
        """아이디 유효성 검사"""
        if not username:
            return False, "아이디를 입력해주세요."
        
        if len(username) < self.MIN_USERNAME_LENGTH:
            return False, f"아이디는 {self.MIN_USERNAME_LENGTH}자 이상이어야 합니다."
        
        if len(username) > self.MAX_USERNAME_LENGTH:
            return False, f"아이디는 {self.MAX_USERNAME_LENGTH}자 이하여야 합니다."
        
        # 영문, 숫자, 언더스코어만 허용
        if not username.replace('_', '').isalnum():
            return False, "아이디는 영문, 숫자, 언더스코어(_)만 사용할 수 있습니다."
        
        # 첫 글자는 영문
        if not username[0].isalpha():
            return False, "아이디는 영문으로 시작해야 합니다."
        
        return True, ""
    
    def validate_password(self, password: str) -> tuple[bool, str]:
        """비밀번호 유효성 검사"""
        if not password:
            return False, "비밀번호를 입력해주세요."
        
        if len(password) < self.MIN_PASSWORD_LENGTH:
            return False, f"비밀번호는 {self.MIN_PASSWORD_LENGTH}자 이상이어야 합니다."
        
        if len(password) > self.MAX_PASSWORD_LENGTH:
            return False, f"비밀번호가 너무 깁니다."
        
        return True, ""
    
    def create_user(self, username: str, password: str) -> tuple[bool, str, Optional[User]]:
        """
        회원가입
        Returns: (성공 여부, 메시지, User 객체)
        """
        # 유효성 검사
        valid, msg = self.validate_username(username)
        if not valid:
            return False, msg, None
        
        valid, msg = self.validate_password(password)
        if not valid:
            return False, msg, None
        
        # 비밀번호 해시
        password_hash, _ = self._hash_password(password)
        
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                
                # 중복 확인
                cursor.execute("SELECT id FROM users WHERE username = ?", (username.lower(),))
                if cursor.fetchone():
                    return False, "이미 사용 중인 아이디입니다.", None
                
                # 사용자 생성
                cursor.execute(
                    "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                    (username.lower(), password_hash)
                )
                user_id = cursor.lastrowid
                
                # 생성된 사용자 조회
                cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
                row = cursor.fetchone()
                
                user = User(
                    id=row['id'],
                    username=row['username'],
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.now(),
                    last_login=None
                )
                
                return True, "회원가입이 완료되었습니다.", user
                
        except Exception as e:
            return False, f"회원가입 중 오류가 발생했습니다: {str(e)}", None
    
    def authenticate(self, username: str, password: str) -> tuple[bool, str, Optional[User]]:
        """
        로그인 인증
        Returns: (성공 여부, 메시지, User 객체)
        """
        if not username or not password:
            return False, "아이디와 비밀번호를 입력해주세요.", None
        
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                
                # 사용자 조회
                cursor.execute(
                    "SELECT * FROM users WHERE username = ?",
                    (username.lower(),)
                )
                row = cursor.fetchone()
                
                if not row:
                    return False, "아이디 또는 비밀번호가 올바르지 않습니다.", None
                
                # 비밀번호 검증
                if not self._verify_password(password, row['password_hash']):
                    return False, "아이디 또는 비밀번호가 올바르지 않습니다.", None
                
                # 마지막 로그인 시간 업데이트
                cursor.execute(
                    "UPDATE users SET last_login = ? WHERE id = ?",
                    (datetime.now().isoformat(), row['id'])
                )
                
                user = User(
                    id=row['id'],
                    username=row['username'],
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.now(),
                    last_login=datetime.now()
                )
                
                return True, "로그인 성공", user
                
        except Exception as e:
            return False, f"로그인 중 오류가 발생했습니다: {str(e)}", None
    
    def change_password(self, user_id: int, current_password: str, new_password: str) -> tuple[bool, str]:
        """
        비밀번호 변경 (본인)
        Returns: (성공 여부, 메시지)
        """
        # 새 비밀번호 유효성 검사
        valid, msg = self.validate_password(new_password)
        if not valid:
            return False, msg
        
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                
                # 현재 비밀번호 확인
                cursor.execute("SELECT password_hash FROM users WHERE id = ?", (user_id,))
                row = cursor.fetchone()
                
                if not row:
                    return False, "사용자를 찾을 수 없습니다."
                
                if not self._verify_password(current_password, row['password_hash']):
                    return False, "현재 비밀번호가 올바르지 않습니다."
                
                # 새 비밀번호 해시
                new_hash, _ = self._hash_password(new_password)
                
                # 비밀번호 업데이트
                cursor.execute(
                    "UPDATE users SET password_hash = ? WHERE id = ?",
                    (new_hash, user_id)
                )
                
                return True, "비밀번호가 변경되었습니다."
                
        except Exception as e:
            return False, f"비밀번호 변경 중 오류가 발생했습니다: {str(e)}"
    
    def reset_password(self, user_id: int, new_password: Optional[str] = None) -> tuple[bool, str, Optional[str]]:
        """
        비밀번호 초기화 (관리자용)
        Returns: (성공 여부, 메시지, 새 비밀번호)
        """
        # 임시 비밀번호 생성 또는 지정된 비밀번호 사용
        if new_password is None:
            new_password = self._generate_temp_password()
        else:
            valid, msg = self.validate_password(new_password)
            if not valid:
                return False, msg, None
        
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                
                # 사용자 확인
                cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
                row = cursor.fetchone()
                
                if not row:
                    return False, "사용자를 찾을 수 없습니다.", None
                
                # 새 비밀번호 해시
                new_hash, _ = self._hash_password(new_password)
                
                # 비밀번호 업데이트
                cursor.execute(
                    "UPDATE users SET password_hash = ? WHERE id = ?",
                    (new_hash, user_id)
                )
                
                return True, f"'{row['username']}' 사용자의 비밀번호가 초기화되었습니다.", new_password
                
        except Exception as e:
            return False, f"비밀번호 초기화 중 오류가 발생했습니다: {str(e)}", None
    
    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """사용자 ID로 조회"""
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                return User(
                    id=row['id'],
                    username=row['username'],
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.now(),
                    last_login=datetime.fromisoformat(row['last_login']) if row['last_login'] else None
                )
        except Exception:
            return None
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """사용자 이름으로 조회"""
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE username = ?", (username.lower(),))
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                return User(
                    id=row['id'],
                    username=row['username'],
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.now(),
                    last_login=datetime.fromisoformat(row['last_login']) if row['last_login'] else None
                )
        except Exception:
            return None
    
    def get_all_users(self) -> List[Dict[str, Any]]:
        """모든 사용자 목록 (관리자용)"""
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, username, created_at, last_login FROM users ORDER BY created_at DESC")
                rows = cursor.fetchall()
                
                users = []
                for row in rows:
                    users.append({
                        "id": row['id'],
                        "username": row['username'],
                        "created_at": row['created_at'],
                        "last_login": row['last_login'],
                    })
                
                return users
        except Exception:
            return []
    
    def delete_user(self, user_id: int) -> tuple[bool, str]:
        """사용자 삭제 (관리자용)"""
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                
                # 사용자 확인
                cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
                row = cursor.fetchone()
                
                if not row:
                    return False, "사용자를 찾을 수 없습니다."
                
                username = row['username']
                
                # 사용자 삭제
                cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
                
                return True, f"'{username}' 사용자가 삭제되었습니다."
                
        except Exception as e:
            return False, f"사용자 삭제 중 오류가 발생했습니다: {str(e)}"
    
    def get_or_create_guest(self) -> tuple[bool, str, Optional[User]]:
        """
        게스트 계정 가져오기 (없으면 생성)
        Returns: (성공 여부, 메시지, User 객체)
        """
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                
                # 게스트 계정 조회
                cursor.execute(
                    "SELECT * FROM users WHERE username = ?",
                    (self.GUEST_USERNAME,)
                )
                row = cursor.fetchone()
                
                if row:
                    # 기존 게스트 계정 반환
                    # 마지막 로그인 시간 업데이트
                    cursor.execute(
                        "UPDATE users SET last_login = ? WHERE id = ?",
                        (datetime.now().isoformat(), row['id'])
                    )
                    
                    user = User(
                        id=row['id'],
                        username=row['username'],
                        created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.now(),
                        last_login=datetime.now()
                    )
                    return True, "게스트로 로그인되었습니다.", user
                
                # 게스트 계정 생성
                password_hash, _ = self._hash_password(self.GUEST_PASSWORD)
                
                cursor.execute(
                    "INSERT INTO users (username, password_hash, last_login) VALUES (?, ?, ?)",
                    (self.GUEST_USERNAME, password_hash, datetime.now().isoformat())
                )
                user_id = cursor.lastrowid
                
                user = User(
                    id=user_id,
                    username=self.GUEST_USERNAME,
                    created_at=datetime.now(),
                    last_login=datetime.now()
                )
                
                return True, "게스트 계정이 생성되었습니다.", user
                
        except Exception as e:
            return False, f"게스트 로그인 중 오류가 발생했습니다: {str(e)}", None
    
    def is_guest(self, user_id: int) -> bool:
        """사용자가 게스트인지 확인"""
        user = self.get_user_by_id(user_id)
        return user is not None and user.username == self.GUEST_USERNAME


# 전역 인스턴스
auth_manager = AuthManager()

