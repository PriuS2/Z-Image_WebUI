"""관리자 API 라우터 (localhost 전용)"""

from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from config.defaults import DEFAULT_GPU_SETTINGS
from utils.settings import settings
from utils.auth import auth_manager
from utils.session import session_manager
from utils.api_keys import api_key_manager
from utils.queue_manager import generation_queue
from utils.qwen_edit import qwen_edit_manager
from utils.gpu_monitor import gpu_monitor
from utils.history import clear_history_manager_cache
from utils.favorites import clear_favorites_manager_cache
from utils.edit_history import clear_edit_history_manager_cache
from services.ws_manager import ws_manager

from routers.dependencies import require_admin


router = APIRouter()


# ============= Pydantic 모델 =============
class ResetPasswordRequest(BaseModel):
    new_password: Optional[str] = None  # None이면 임시 비밀번호 자동 생성


class GPUSettingsRequest(BaseModel):
    generation_gpu: Optional[str] = None  # "auto", "cuda:0", "cuda:1", "cpu"
    edit_gpu: Optional[str] = None        # "auto", "cuda:0", "cuda:1", "cpu"


class CreateAPIKeyRequest(BaseModel):
    """API 키 생성 요청"""
    name: str


class UpdateAPIKeyRequest(BaseModel):
    """API 키 수정 요청"""
    name: Optional[str] = None
    is_active: Optional[bool] = None


# ============= 유틸리티 함수 =============
def _format_bytes(total_size: int) -> str:
    """바이트를 사람이 읽기 쉬운 단위로 변환"""
    if total_size < 1024:
        return f"{total_size} B"
    if total_size < 1024 * 1024:
        return f"{total_size / 1024:.1f} KB"
    if total_size < 1024 * 1024 * 1024:
        return f"{total_size / (1024 * 1024):.1f} MB"
    return f"{total_size / (1024 * 1024 * 1024):.2f} GB"


def _get_data_size_by_data_id(data_id: str) -> str:
    """data_id(user_{id}) 기준 데이터 크기 계산 (세션 화면용)"""
    from config.defaults import DATA_DIR, OUTPUTS_DIR
    total_size = 0
    sessions_dir = DATA_DIR / "sessions" / data_id
    outputs_dir = OUTPUTS_DIR / data_id

    for d in (sessions_dir, outputs_dir):
        if d.exists():
            for f in d.rglob("*"):
                if f.is_file():
                    try:
                        total_size += f.stat().st_size
                    except Exception:
                        pass
    return _format_bytes(total_size)


def _parse_user_id_from_data_id(data_id: str) -> Optional[int]:
    """user_123 -> 123"""
    if not isinstance(data_id, str):
        return None
    if not data_id.startswith("user_"):
        return None
    try:
        return int(data_id.split("_", 1)[1])
    except Exception:
        return None


# ============= 사용자 관리 엔드포인트 =============
@router.get("/users")
async def get_all_users(request: Request):
    """모든 사용자 목록 (관리자 전용)"""
    require_admin(request)
    
    users = auth_manager.get_all_users()
    return {"users": users}


@router.post("/users/{user_id}/reset-password")
async def admin_reset_password(request: Request, user_id: int, data: ResetPasswordRequest):
    """사용자 비밀번호 초기화 (관리자 전용)"""
    require_admin(request)
    
    success, message, new_password = auth_manager.reset_password(user_id, data.new_password)
    
    if not success:
        raise HTTPException(400, message)
    
    return {
        "success": True,
        "message": message,
        "new_password": new_password  # 관리자에게 임시 비밀번호 표시
    }


@router.delete("/users/{user_id}")
async def admin_delete_user(request: Request, user_id: int):
    """사용자 삭제 (관리자 전용)"""
    require_admin(request)
    
    # 사용자 삭제
    success, message = auth_manager.delete_user(user_id)
    
    if not success:
        raise HTTPException(400, message)
    
    # 사용자 데이터도 삭제
    await session_manager.delete_user_data(user_id)
    
    return {
        "success": True,
        "message": message
    }


@router.delete("/users/{user_id}/data")
async def admin_delete_user_data(request: Request, user_id: int):
    """사용자 데이터 삭제 (계정 유지, 관리자 전용)

    - 계정(DB)은 유지
    - 히스토리/즐겨찾기/편집 히스토리/설정 등 세션 데이터 폴더 삭제
    - 생성/편집 결과 이미지 등 outputs 폴더 삭제
    """
    require_admin(request)

    data_id = f"user_{user_id}"

    # 진행 중/대기 중 작업이 있으면 먼저 제거(파일 재생성/경합 방지)
    await generation_queue.remove_session_items(data_id)

    # 사용자 데이터 삭제(세션 폴더 + outputs 폴더)
    await session_manager.delete_user_data(user_id)

    # 캐시 제거: 삭제 직후 기존 캐시가 다시 파일을 저장하는 것을 방지
    clear_history_manager_cache(data_id)
    clear_favorites_manager_cache(data_id)
    clear_edit_history_manager_cache(data_id)

    return {
        "success": True,
        "message": "사용자 데이터가 삭제되었습니다. (계정은 유지됩니다.)",
        "data_id": data_id,
    }


# ============= 세션 관리 엔드포인트 =============
@router.get("/sessions")
async def get_all_sessions(request: Request):
    """사용자(계정) 세션/접속 현황 목록 (관리자 전용)

    기존에는 WebSocket '연결된' 계정만 반환했지만,
    UI(세션 관리)에서 등록된 계정 전체가 보이길 기대하는 경우가 많아
    등록된 사용자 목록 + 연결 여부를 함께 내려준다.
    """
    require_admin(request)

    connected_keys = set(ws_manager.get_connected_keys())

    rows: List[Dict[str, Any]] = []

    # 1) 등록된 사용자 전체를 포함
    all_users = auth_manager.get_all_users()
    for u in all_users:
        user_id = u.get("id")
        if user_id is None:
            continue

        data_id = f"user_{user_id}"
        connected = data_id in connected_keys

        # last_activity: 세션 매니저의 활동 시간(있으면) → 없으면 last_login(있으면)
        session = session_manager.get_session_by_user(user_id)
        last_activity_ts: Optional[float] = None
        last_activity_iso: Optional[str] = None
        if session:
            last_activity_ts = session.last_activity
            last_activity_iso = datetime.fromtimestamp(session.last_activity).isoformat()
        else:
            # auth_manager.get_all_users()는 last_login을 ISO 문자열로 내려줌 (또는 None)
            last_login = u.get("last_login")
            if last_login:
                last_activity_iso = last_login

        rows.append({
            "data_id": data_id,
            "user_id": user_id,
            "username": u.get("username"),
            "last_activity": last_activity_iso,
            "data_size": _get_data_size_by_data_id(data_id),
            "connected": connected,
            "_sort_ts": last_activity_ts,  # 정렬용(응답에는 제거)
        })

    # 2) 예외적으로 연결은 있는데 DB에 없는 키가 있으면 추가로 노출(디버깅/운영 편의)
    known_data_ids = {r["data_id"] for r in rows if r.get("data_id")}
    for data_id in sorted(connected_keys):
        if data_id in known_data_ids:
            continue
        user_id = _parse_user_id_from_data_id(data_id)
        session = session_manager.get_session_by_user(user_id) if user_id is not None else None
        last_activity_iso = datetime.fromtimestamp(session.last_activity).isoformat() if session else None

        rows.append({
            "data_id": data_id,
            "user_id": user_id,
            "username": None,
            "last_activity": last_activity_iso,
            "data_size": _get_data_size_by_data_id(data_id),
            "connected": True,
            "_sort_ts": session.last_activity if session else None,
        })

    # 정렬: 연결된 계정 우선, 최근 활동 우선
    rows.sort(key=lambda r: (bool(r.get("connected")), r.get("_sort_ts") or 0), reverse=True)
    for r in rows:
        r.pop("_sort_ts", None)

    return {"users": rows}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, request: Request):
    """사용자(계정) 접속 종료/정리 (관리자 전용)"""
    require_admin(request)

    # 프론트에서 넘어오는 값은 이제 data_id(user_{id})로 사용
    data_id = session_id
    user_id = _parse_user_id_from_data_id(data_id)

    # WebSocket 강제 종료 + 대기열 제거
    closed = await ws_manager.disconnect_key(data_id)
    await generation_queue.remove_session_items(data_id)

    # 세션 매핑 정리(가능한 경우)
    if user_id is not None:
        existing = session_manager.get_session_by_user(user_id)
        if existing:
            await session_manager.delete_session(existing.session_id)

    return {"success": True, "closed_connections": closed}


# ============= GPU 관리 엔드포인트 =============
@router.get("/gpu-status")
async def get_gpu_status(request: Request):
    """GPU 상태 조회 (관리자 전용)"""
    require_admin(request)
    
    # 전역 상태 접근 (지연 import)
    import app as app_module
    
    gpu_info = gpu_monitor.get_system_info()
    
    # 현재 모델 상태 추가
    gpu_info["models"] = {
        "generation": {
            "loaded": app_module.pipe is not None,
            "name": app_module.current_model,
            "device": app_module.device,
        },
        "edit": {
            "loaded": qwen_edit_manager.is_loaded,
            "name": qwen_edit_manager.current_model,
            "device": qwen_edit_manager.device,
            "quantization": "NF4 (4bit)",  # Qwen은 4bit NF4 고정
            "cpu_offload": qwen_edit_manager.cpu_offload_enabled,
        }
    }
    
    # 현재 GPU 설정 추가
    gpu_info["current_settings"] = {
        "generation_gpu": settings.get("generation_gpu", DEFAULT_GPU_SETTINGS["generation_gpu"]),
        "edit_gpu": settings.get("edit_gpu", DEFAULT_GPU_SETTINGS["edit_gpu"]),
    }
    
    return gpu_info


@router.post("/gpu-settings")
async def update_gpu_settings(request: Request, gpu_settings: GPUSettingsRequest):
    """GPU 설정 업데이트 (관리자 전용)"""
    require_admin(request)
    
    # 유효한 디바이스인지 확인
    available_devices = gpu_monitor.get_available_devices()
    
    if gpu_settings.generation_gpu is not None:
        if gpu_settings.generation_gpu not in available_devices:
            raise HTTPException(400, f"유효하지 않은 디바이스: {gpu_settings.generation_gpu}")
        settings.set("generation_gpu", gpu_settings.generation_gpu)
    
    if gpu_settings.edit_gpu is not None:
        if gpu_settings.edit_gpu not in available_devices:
            raise HTTPException(400, f"유효하지 않은 디바이스: {gpu_settings.edit_gpu}")
        settings.set("edit_gpu", gpu_settings.edit_gpu)
    
    return {
        "success": True,
        "message": "GPU 설정이 업데이트되었습니다.",
        "settings": {
            "generation_gpu": settings.get("generation_gpu", DEFAULT_GPU_SETTINGS["generation_gpu"]),
            "edit_gpu": settings.get("edit_gpu", DEFAULT_GPU_SETTINGS["edit_gpu"]),
        }
    }


@router.get("/available-devices")
async def get_available_devices(request: Request):
    """사용 가능한 디바이스 목록 (관리자 전용)"""
    require_admin(request)
    
    return {
        "devices": gpu_monitor.get_available_devices(),
        "gpu_count": gpu_monitor.gpu_count,
        "cuda_available": gpu_monitor.cuda_available,
    }


# ============= API 키 관리 엔드포인트 =============
@router.get("/api-keys")
async def list_api_keys(request: Request):
    """API 키 목록 조회 (관리자 전용)"""
    require_admin(request)
    
    keys = api_key_manager.list_api_keys()
    return {"api_keys": keys}


@router.post("/api-keys")
async def create_api_key(request: Request, data: CreateAPIKeyRequest):
    """새 API 키 생성 (관리자 전용)"""
    require_admin(request)
    
    success, message, full_key, api_key_obj = api_key_manager.create_api_key(data.name)
    
    if not success:
        raise HTTPException(400, message)
    
    return {
        "success": True,
        "message": message,
        "api_key": full_key,  # 전체 키는 생성 시에만 반환
        "key_info": api_key_obj.to_dict() if api_key_obj else None
    }


@router.patch("/api-keys/{key_id}")
async def update_api_key(request: Request, key_id: int, data: UpdateAPIKeyRequest):
    """API 키 수정 (관리자 전용)"""
    require_admin(request)
    
    success, message = api_key_manager.update_api_key(
        key_id, 
        name=data.name, 
        is_active=data.is_active
    )
    
    if not success:
        raise HTTPException(400, message)
    
    return {
        "success": True,
        "message": message
    }


@router.delete("/api-keys/{key_id}")
async def delete_api_key(request: Request, key_id: int):
    """API 키 삭제 (관리자 전용)"""
    require_admin(request)
    
    success, message = api_key_manager.delete_api_key(key_id)
    
    if not success:
        raise HTTPException(400, message)
    
    return {
        "success": True,
        "message": message
    }
