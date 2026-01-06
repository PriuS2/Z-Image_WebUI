"""공통 라우터 의존성"""

from typing import Optional, Dict, Any

from fastapi import HTTPException, Request, Response, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from utils.session import session_manager, is_localhost, SessionManager, SessionInfo
from utils.auth import auth_manager
from utils.api_keys import api_key_manager


async def get_session_from_request(request: Request, create_if_missing: bool = False) -> Optional[SessionInfo]:
    """
    요청에서 세션 가져오기
    - 기본: **비로그인(쿠키 없음/유효하지 않음/메모리에 없음)에서는 세션을 생성하지 않음**
    - 로그인/회원가입 등 일부 엔드포인트에서만 create_if_missing=True로 세션 생성
    """
    session_id = request.cookies.get(SessionManager.COOKIE_NAME)

    # 쿠키가 있고 메모리에 살아있는 세션이면 반환
    if session_id and session_manager.validate_session_id(session_id):
        session = session_manager.get_session(session_id)
        if session:
            session.update_activity()
            return session

    # 필요한 경우에만 새 세션 생성
    if create_if_missing:
        return await session_manager.get_or_create_session(session_id)

    return None


def require_auth(session: Optional[SessionInfo]) -> None:
    """인증 필수 체크 - 로그인하지 않으면 예외 발생"""
    if not session or not session.is_authenticated:
        raise HTTPException(401, "로그인이 필요합니다.")


# API 키 인증을 위한 보안 스키마 (Swagger docs에서 사용)
api_key_scheme = HTTPBearer(
    scheme_name="API Key",
    description="API 키를 입력하세요 (zimg_로 시작하는 키)",
    auto_error=False  # 인증 실패 시 자동 에러 발생 안 함 (세션 인증 폴백 허용)
)


async def get_api_key_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(api_key_scheme)
) -> Optional[str]:
    """Swagger docs에서 API 키 인증을 위한 의존성"""
    if credentials:
        return credentials.credentials
    return None


def get_api_key_from_request(request: Request) -> Optional[str]:
    """요청에서 API 키 추출 (Authorization: Bearer <api_key>)"""
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[7:]
    return None


async def get_auth_from_request(request: Request) -> Dict[str, Any]:
    """
    요청에서 인증 정보 가져오기 (API 키 또는 세션)
    
    Returns:
        {"type": "api_key", "api_key": APIKey} 또는
        {"type": "session", "session": SessionInfo} 또는
        예외 발생
    """
    # 1. Authorization 헤더에서 API 키 확인
    api_key_str = get_api_key_from_request(request)
    if api_key_str:
        is_valid, api_key_obj = api_key_manager.validate_api_key(api_key_str)
        if is_valid and api_key_obj:
            return {"type": "api_key", "api_key": api_key_obj}
        raise HTTPException(401, "유효하지 않은 API 키입니다.")
    
    # 2. 기존 세션 인증으로 폴백
    session = await get_session_from_request(request)
    if session and session.is_authenticated:
        return {"type": "session", "session": session}
    
    raise HTTPException(401, "인증이 필요합니다. 로그인하거나 API 키를 사용하세요.")


async def require_auth_or_api_key(request: Request) -> Dict[str, Any]:
    """인증 필수 체크 - 세션 또는 API 키 중 하나가 있어야 함"""
    return await get_auth_from_request(request)


def require_admin(request: Request) -> None:
    """관리자 권한 체크 - localhost가 아니면 예외 발생"""
    client_host = request.client.host if request.client else None
    if not is_localhost(client_host):
        raise HTTPException(403, "관리자 권한이 필요합니다.")


def set_session_cookie(response: Response, session: Optional[SessionInfo]):
    """응답에 세션 쿠키 설정"""
    if not session:
        return
    response.set_cookie(
        key=SessionManager.COOKIE_NAME,
        value=session.session_id,
        max_age=SessionManager.COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax"
    )


def clear_session_cookie(response: Response):
    """세션 쿠키 제거"""
    response.delete_cookie(key=SessionManager.COOKIE_NAME)
