"""히스토리 및 즐겨찾기 API 라우터"""

from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from utils.history import get_history_manager_sync
from utils.favorites import get_favorites_manager_sync

from routers.dependencies import (
    get_session_from_request,
    require_auth,
    set_session_cookie,
)


router = APIRouter()


# ============= Pydantic 모델 =============
class ConversationUpdateRequest(BaseModel):
    conversation: List[Dict[str, Any]]


class FavoriteRequest(BaseModel):
    name: str
    prompt: str
    settings: dict = {}


# ============= 히스토리 엔드포인트 =============
@router.get("/history")
async def get_history(request: Request):
    """히스토리 목록 (사용자별)"""
    session = await get_session_from_request(request)
    require_auth(session)
    history_mgr = get_history_manager_sync(session.data_id)
    entries = history_mgr.get_all()
    
    response = JSONResponse(content={"history": [e.to_dict() for e in entries[:50]]})
    set_session_cookie(response, session)
    return response


@router.get("/history/{history_id}")
async def get_history_detail(history_id: str, request: Request):
    """히스토리 상세 정보 (사용자별)"""
    session = await get_session_from_request(request)
    require_auth(session)
    history_mgr = get_history_manager_sync(session.data_id)
    entry = history_mgr.get_by_id(history_id)
    
    if not entry:
        raise HTTPException(404, "히스토리를 찾을 수 없습니다.")
    
    return {"history": entry.to_dict()}


@router.patch("/history/{history_id}/conversation")
async def update_history_conversation(history_id: str, request: Request, conv_request: ConversationUpdateRequest):
    """히스토리의 대화 내용 업데이트 (사용자별)"""
    session = await get_session_from_request(request)
    require_auth(session)
    history_mgr = get_history_manager_sync(session.data_id)
    entry = history_mgr.get_by_id(history_id)
    
    if not entry:
        raise HTTPException(404, "히스토리를 찾을 수 없습니다.")
    
    entry.conversation = conv_request.conversation
    history_mgr._save()
    
    return {"success": True}


@router.delete("/history")
async def clear_history(request: Request):
    """히스토리 삭제 (사용자별)"""
    session = await get_session_from_request(request)
    require_auth(session)
    history_mgr = get_history_manager_sync(session.data_id)
    history_mgr.clear()
    return {"success": True}


# ============= 즐겨찾기 엔드포인트 =============
@router.get("/favorites")
async def get_favorites(request: Request):
    """즐겨찾기 목록 (사용자별)"""
    session = await get_session_from_request(request)
    require_auth(session)
    fav_mgr = get_favorites_manager_sync(session.data_id)
    entries = fav_mgr.get_all()
    
    response = JSONResponse(content={"favorites": [e.to_dict() for e in entries]})
    set_session_cookie(response, session)
    return response


@router.post("/favorites")
async def add_favorite(request: Request, fav_request: FavoriteRequest):
    """즐겨찾기 추가 (사용자별)"""
    session = await get_session_from_request(request)
    require_auth(session)
    fav_mgr = get_favorites_manager_sync(session.data_id)
    entry = fav_mgr.add(
        name=fav_request.name,
        prompt=fav_request.prompt,
        settings=fav_request.settings
    )
    return {"success": True, "id": entry.id}


@router.delete("/favorites/{fav_id}")
async def delete_favorite(fav_id: str, request: Request):
    """즐겨찾기 삭제 (사용자별)"""
    session = await get_session_from_request(request)
    require_auth(session)
    fav_mgr = get_favorites_manager_sync(session.data_id)
    success = fav_mgr.delete(fav_id)
    return {"success": success}
