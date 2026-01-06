"""인증 관련 API 라우터"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from utils.auth import auth_manager
from utils.session import session_manager, is_localhost

from routers.dependencies import (
    get_session_from_request,
    require_auth,
    set_session_cookie,
    clear_session_cookie,
)


router = APIRouter()


# ============= Pydantic 모델 =============
class RegisterRequest(BaseModel):
    username: str
    password: str
    password_confirm: str


class LoginRequest(BaseModel):
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str
    new_password_confirm: str


# ============= 엔드포인트 =============
@router.post("/register")
async def register(request: Request, data: RegisterRequest):
    """회원가입"""
    # 회원가입은 세션(로그인 쿠키) 발급이 필요하므로 생성 허용
    session = await get_session_from_request(request, create_if_missing=True)
    
    # 비밀번호 확인
    if data.password != data.password_confirm:
        raise HTTPException(400, "비밀번호가 일치하지 않습니다.")
    
    # 회원가입
    success, message, user = auth_manager.create_user(data.username, data.password)
    
    if not success:
        raise HTTPException(400, message)
    
    response = JSONResponse(content={
        "success": True,
        "message": message,
        "user": user.to_dict() if user else None
    })
    set_session_cookie(response, session)
    return response


@router.post("/login")
async def login(request: Request, data: LoginRequest):
    """로그인"""
    # 로그인은 세션(로그인 쿠키) 발급이 필요하므로 생성 허용
    session = await get_session_from_request(request, create_if_missing=True)
    
    # 아이디와 비밀번호가 모두 비어있으면 게스트로 로그인
    if not data.username.strip() and not data.password.strip():
        success, message, user = auth_manager.get_or_create_guest()
    else:
        # 일반 인증
        success, message, user = auth_manager.authenticate(data.username, data.password)
    
    if not success or not user:
        raise HTTPException(401, message)
    
    # 세션에 로그인 정보 연결
    await session_manager.login_session(session.session_id, user.id, user.username)
    
    response = JSONResponse(content={
        "success": True,
        "message": message,
        "user": user.to_dict(),
        "is_guest": user.username == auth_manager.GUEST_USERNAME
    })
    set_session_cookie(response, session)
    return response


@router.post("/guest")
async def guest_login(request: Request):
    """게스트 로그인"""
    # 게스트 로그인은 세션(로그인 쿠키) 발급이 필요하므로 생성 허용
    session = await get_session_from_request(request, create_if_missing=True)
    
    # 게스트 계정 가져오기 (없으면 생성)
    success, message, user = auth_manager.get_or_create_guest()
    
    if not success or not user:
        raise HTTPException(500, message)
    
    # 세션에 로그인 정보 연결
    await session_manager.login_session(session.session_id, user.id, user.username)
    
    response = JSONResponse(content={
        "success": True,
        "message": message,
        "user": user.to_dict(),
        "is_guest": True
    })
    set_session_cookie(response, session)
    return response


@router.post("/logout")
async def logout(request: Request):
    """로그아웃"""
    session = await get_session_from_request(request)
    
    if session and session.is_authenticated:
        await session_manager.logout_session(session.session_id)
    
    response = JSONResponse(content={
        "success": True,
        "message": "로그아웃되었습니다."
    })
    # 로그아웃은 쿠키 제거
    clear_session_cookie(response)
    return response


@router.get("/me")
async def get_current_user(request: Request):
    """현재 로그인된 사용자 정보"""
    session = await get_session_from_request(request)
    
    # 관리자 여부 확인
    client_host = request.client.host if request.client else None
    is_admin = is_localhost(client_host)
    
    if not session or not session.is_authenticated:
        response = JSONResponse(content={
            "authenticated": False,
            "user": None,
            "is_admin": is_admin
        })
        clear_session_cookie(response)
    else:
        user = auth_manager.get_user_by_id(session.user_id)
        response = JSONResponse(content={
            "authenticated": True,
            "user": user.to_dict() if user else {
                "id": session.user_id,
                "username": session.username
            },
            "is_admin": is_admin
        })
        set_session_cookie(response, session)
    return response


@router.post("/change-password")
async def change_password(request: Request, data: ChangePasswordRequest):
    """비밀번호 변경 (본인)"""
    session = await get_session_from_request(request)
    require_auth(session)
    
    # 비밀번호 확인
    if data.new_password != data.new_password_confirm:
        raise HTTPException(400, "새 비밀번호가 일치하지 않습니다.")
    
    # 비밀번호 변경
    success, message = auth_manager.change_password(
        session.user_id,
        data.current_password,
        data.new_password
    )
    
    if not success:
        raise HTTPException(400, message)
    
    response = JSONResponse(content={
        "success": True,
        "message": message
    })
    set_session_cookie(response, session)
    return response
