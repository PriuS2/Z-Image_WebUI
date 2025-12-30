"""API 키 관리자 - API 키 생성, 검증, 관리"""

import hashlib
import secrets
import string
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from utils.database import get_db


@dataclass
class APIKey:
    """API 키 정보"""
    id: int
    name: str
    key_prefix: str
    created_at: datetime
    last_used: Optional[datetime] = None
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "key_prefix": self.key_prefix,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "is_active": self.is_active,
        }


class APIKeyManager:
    """API 키 관리자"""
    
    # API 키 형식: zimg_ + 40자리 랜덤 문자열
    KEY_PREFIX = "zimg_"
    KEY_LENGTH = 40
    
    def __init__(self):
        pass
    
    def _generate_key(self) -> str:
        """새 API 키 생성"""
        characters = string.ascii_letters + string.digits
        random_part = ''.join(secrets.choice(characters) for _ in range(self.KEY_LENGTH))
        return f"{self.KEY_PREFIX}{random_part}"
    
    def _hash_key(self, api_key: str) -> str:
        """API 키 해시 (SHA-256)"""
        return hashlib.sha256(api_key.encode('utf-8')).hexdigest()
    
    def _get_display_prefix(self, api_key: str) -> str:
        """표시용 접두사 생성 (zimg_xxxx...)"""
        if api_key.startswith(self.KEY_PREFIX):
            # zimg_ + 앞 4자리 + ...
            return api_key[:9] + "..."
        return api_key[:8] + "..."
    
    def create_api_key(self, name: str) -> tuple[bool, str, Optional[str], Optional[APIKey]]:
        """
        새 API 키 생성
        
        Args:
            name: API 키 이름/설명
            
        Returns:
            (성공 여부, 메시지, 전체 API 키 (최초 1회만), APIKey 객체)
        """
        if not name or not name.strip():
            return False, "API 키 이름을 입력해주세요.", None, None
        
        name = name.strip()
        if len(name) > 100:
            return False, "API 키 이름은 100자 이하여야 합니다.", None, None
        
        # 새 키 생성
        api_key = self._generate_key()
        key_hash = self._hash_key(api_key)
        key_prefix = self._get_display_prefix(api_key)
        
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                
                # API 키 저장
                cursor.execute(
                    """INSERT INTO api_keys (name, key_hash, key_prefix, is_active) 
                       VALUES (?, ?, ?, 1)""",
                    (name, key_hash, key_prefix)
                )
                key_id = cursor.lastrowid
                
                # 생성된 키 조회
                cursor.execute("SELECT * FROM api_keys WHERE id = ?", (key_id,))
                row = cursor.fetchone()
                
                api_key_obj = APIKey(
                    id=row['id'],
                    name=row['name'],
                    key_prefix=row['key_prefix'],
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.now(),
                    last_used=None,
                    is_active=bool(row['is_active'])
                )
                
                return True, "API 키가 생성되었습니다.", api_key, api_key_obj
                
        except Exception as e:
            return False, f"API 키 생성 중 오류가 발생했습니다: {str(e)}", None, None
    
    def validate_api_key(self, api_key: str) -> tuple[bool, Optional[APIKey]]:
        """
        API 키 검증
        
        Args:
            api_key: 검증할 API 키
            
        Returns:
            (유효 여부, APIKey 객체)
        """
        if not api_key or not api_key.startswith(self.KEY_PREFIX):
            return False, None
        
        key_hash = self._hash_key(api_key)
        
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                
                # 해시로 키 조회
                cursor.execute(
                    "SELECT * FROM api_keys WHERE key_hash = ? AND is_active = 1",
                    (key_hash,)
                )
                row = cursor.fetchone()
                
                if not row:
                    return False, None
                
                # 마지막 사용 시간 업데이트
                cursor.execute(
                    "UPDATE api_keys SET last_used = ? WHERE id = ?",
                    (datetime.now().isoformat(), row['id'])
                )
                
                api_key_obj = APIKey(
                    id=row['id'],
                    name=row['name'],
                    key_prefix=row['key_prefix'],
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.now(),
                    last_used=datetime.now(),
                    is_active=bool(row['is_active'])
                )
                
                return True, api_key_obj
                
        except Exception:
            return False, None
    
    def list_api_keys(self) -> List[Dict[str, Any]]:
        """모든 API 키 목록 조회 (해시 제외)"""
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """SELECT id, name, key_prefix, created_at, last_used, is_active 
                       FROM api_keys ORDER BY created_at DESC"""
                )
                rows = cursor.fetchall()
                
                keys = []
                for row in rows:
                    keys.append({
                        "id": row['id'],
                        "name": row['name'],
                        "key_prefix": row['key_prefix'],
                        "created_at": row['created_at'],
                        "last_used": row['last_used'],
                        "is_active": bool(row['is_active']),
                    })
                
                return keys
        except Exception:
            return []
    
    def get_api_key_by_id(self, key_id: int) -> Optional[APIKey]:
        """ID로 API 키 조회"""
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM api_keys WHERE id = ?", (key_id,))
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                return APIKey(
                    id=row['id'],
                    name=row['name'],
                    key_prefix=row['key_prefix'],
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.now(),
                    last_used=datetime.fromisoformat(row['last_used']) if row['last_used'] else None,
                    is_active=bool(row['is_active'])
                )
        except Exception:
            return None
    
    def update_api_key(self, key_id: int, name: Optional[str] = None, is_active: Optional[bool] = None) -> tuple[bool, str]:
        """
        API 키 정보 수정
        
        Args:
            key_id: API 키 ID
            name: 새 이름 (None이면 변경 안 함)
            is_active: 활성화 상태 (None이면 변경 안 함)
            
        Returns:
            (성공 여부, 메시지)
        """
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                
                # 키 확인
                cursor.execute("SELECT id, name FROM api_keys WHERE id = ?", (key_id,))
                row = cursor.fetchone()
                
                if not row:
                    return False, "API 키를 찾을 수 없습니다."
                
                # 업데이트할 필드 구성
                updates = []
                params = []
                
                if name is not None:
                    name = name.strip()
                    if not name:
                        return False, "API 키 이름을 입력해주세요."
                    if len(name) > 100:
                        return False, "API 키 이름은 100자 이하여야 합니다."
                    updates.append("name = ?")
                    params.append(name)
                
                if is_active is not None:
                    updates.append("is_active = ?")
                    params.append(1 if is_active else 0)
                
                if not updates:
                    return False, "변경할 항목이 없습니다."
                
                # 업데이트 실행
                params.append(key_id)
                cursor.execute(
                    f"UPDATE api_keys SET {', '.join(updates)} WHERE id = ?",
                    tuple(params)
                )
                
                return True, "API 키가 수정되었습니다."
                
        except Exception as e:
            return False, f"API 키 수정 중 오류가 발생했습니다: {str(e)}"
    
    def delete_api_key(self, key_id: int) -> tuple[bool, str]:
        """
        API 키 삭제
        
        Args:
            key_id: 삭제할 API 키 ID
            
        Returns:
            (성공 여부, 메시지)
        """
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                
                # 키 확인
                cursor.execute("SELECT id, name FROM api_keys WHERE id = ?", (key_id,))
                row = cursor.fetchone()
                
                if not row:
                    return False, "API 키를 찾을 수 없습니다."
                
                key_name = row['name']
                
                # 삭제
                cursor.execute("DELETE FROM api_keys WHERE id = ?", (key_id,))
                
                return True, f"API 키 '{key_name}'가 삭제되었습니다."
                
        except Exception as e:
            return False, f"API 키 삭제 중 오류가 발생했습니다: {str(e)}"
    
    def revoke_api_key(self, key_id: int) -> tuple[bool, str]:
        """API 키 비활성화"""
        return self.update_api_key(key_id, is_active=False)
    
    def activate_api_key(self, key_id: int) -> tuple[bool, str]:
        """API 키 활성화"""
        return self.update_api_key(key_id, is_active=True)


# 전역 인스턴스
api_key_manager = APIKeyManager()

