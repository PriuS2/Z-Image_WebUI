"""Z-Image WebUI - FastAPI 기반 대화형 이미지 생성 웹앱 (다중 사용자 지원)"""

import os
import sys
import json
import asyncio
import base64
import random
import gc
import time
import inspect
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from io import BytesIO
from contextlib import asynccontextmanager

# 프로젝트 루트를 path에 추가
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form, Response, Cookie, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.requests import Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn

import torch
from PIL import Image

# 로컬 모듈
from config.defaults import (
    QUANTIZATION_OPTIONS,
    RESOLUTION_PRESETS,
    OUTPUTS_DIR,
    MODELS_DIR,
    SERVER_HOST,
    SERVER_PORT,
    SERVER_RELOAD,
    QWEN_EDIT_AUTO_UNLOAD_TIMEOUT,
    QWEN_EDIT_MODEL_VRAM,
    DEFAULT_QWEN_EDIT_SETTINGS,
    DEFAULT_GPU_SETTINGS,
)
from config.templates import PROMPT_TEMPLATES
from utils.settings import settings
from utils.translator import translator
from utils.prompt_enhancer import prompt_enhancer
from utils.metadata import ImageMetadata, filename_generator
from utils.history import get_history_manager_sync, HistoryManager, clear_history_manager_cache
from utils.favorites import get_favorites_manager_sync, FavoritesManager, clear_favorites_manager_cache
from utils.session import session_manager, is_localhost, SessionManager, SessionInfo
from utils.queue_manager import generation_queue, GenerationQueueManager
from utils.auth import auth_manager, User
from utils.api_keys import api_key_manager, APIKey
from utils.qwen_edit import qwen_edit_manager
from utils.edit_history import get_edit_history_manager_sync, EditHistoryManager, clear_edit_history_manager_cache
from utils.edit_llm import edit_translator, edit_enhancer, edit_suggester
from utils.gpu_monitor import gpu_monitor
from services.ws_manager import ws_manager


# ============= 전역 변수 =============
pipe = None
current_model = None
device = None
last_activity_time = time.time()  # 마지막 활동 시간
auto_unload_task = None  # 자동 언로드 체크 태스크
model_lock = asyncio.Lock()  # 모델 로드/언로드 잠금

# Qwen-Image-Edit 관련
edit_last_activity_time = time.time()  # 편집 모델 마지막 활동 시간
edit_auto_unload_task = None  # 편집 모델 자동 언로드 태스크
edit_model_lock = asyncio.Lock()  # 편집 모델 로드/언로드 잠금


# ============= 모델별 예상 VRAM 사용량 (GB) =============
# 생성 모델: 양자화에 따라 다름
GENERATION_MODEL_VRAM = {
    "BF16 (기본, 최고품질)": 14.0,
    "GGUF Q8_0 (7.22GB, 고품질)": 7.5,
    "GGUF Q6_K (5.91GB, 고품질)": 6.0,
    "GGUF Q5_K_M (5.52GB, 균형)": 5.8,
    "GGUF Q5_K_S (5.19GB, 균형)": 5.5,
    "GGUF Q4_K_M (4.98GB, 추천)": 5.2,
    "GGUF Q4_K_S (4.66GB, 경량)": 4.9,
    "GGUF Q3_K_M (4.12GB, 저사양)": 4.4,
    "GGUF Q3_K_S (3.79GB, 최저사양)": 4.1,
}


# ============= 자동 모델 로드/언로드 함수 =============
async def unload_generation_model_internal():
    """생성 모델 내부 언로드 (lock 없이)"""
    global pipe, current_model, device
    
    if pipe is None:
        return
    
    print("[*] Auto-unloading generation model...")
    
    # GPU 모니터에서 모델 등록 해제
    gpu_monitor.unregister_model("Z-Image-Turbo")
    
    del pipe
    pipe = None
    old_model = current_model
    current_model = None
    
    # GPU 캐시 정리
    gpu_monitor.clear_cache(device)
    gc.collect()
    
    # 클라이언트에게 알림
    await ws_manager.broadcast({
        "type": "model_status_change",
        "model_loaded": False,
        "current_model": None
    })
    await ws_manager.broadcast({
        "type": "system",
        "content": f"🔄 VRAM 확보를 위해 생성 모델({old_model})이 자동 언로드되었습니다."
    })
    
    print(f"[OK] Generation model auto-unloaded. VRAM: {get_vram_info()}")


async def unload_edit_model_internal():
    """편집 모델 내부 언로드 (lock 없이)"""
    if not qwen_edit_manager.is_loaded:
        return
    
    print("[*] Auto-unloading edit model...")
    
    old_model = qwen_edit_manager.current_model
    
    # 편집 모델 언로드 (내부 lock 사용)
    success, message = await qwen_edit_manager.unload_model()
    
    if success:
        # 클라이언트에게 알림
        await ws_manager.broadcast({
            "type": "edit_model_status_change",
            "model_loaded": False,
            "current_model": None
        })
        await ws_manager.broadcast({
            "type": "system",
            "content": f"🔄 VRAM 확보를 위해 편집 모델({old_model})이 자동 언로드되었습니다."
        })
        
        print(f"[OK] Edit model auto-unloaded. VRAM: {get_vram_info()}")


async def ensure_generation_model_loaded(session_id: str = None) -> tuple[bool, str]:
    """
    생성 모델이 로드되어 있는지 확인하고, 없으면 자동 로드
    VRAM이 부족하면 편집 모델을 먼저 언로드
    
    Args:
        session_id: 메시지를 보낼 세션 ID (None이면 broadcast)
    
    Returns:
        (success, message)
    """
    global pipe, current_model, device, model_lock
    
    # 이미 로드되어 있으면 바로 반환
    if pipe is not None:
        return True, "모델이 이미 로드되어 있습니다."
    
    # 모델 잠금 확인
    if model_lock.locked():
        return False, "다른 사용자가 모델을 로드/언로드 중입니다. 잠시 후 다시 시도해주세요."
    
    async with model_lock:
        # 다시 확인 (lock 대기 중 로드되었을 수 있음)
        if pipe is not None:
            return True, "모델이 이미 로드되어 있습니다."
        
        # 설정에서 양자화 옵션 가져오기
        quantization = settings.get("quantization", "BF16 (기본, 최고품질)")
        cpu_offload = settings.get("cpu_offload", False)
        target_device_setting = settings.get("generation_gpu", DEFAULT_GPU_SETTINGS["generation_gpu"])
        
        # 필요한 VRAM 계산
        required_vram = GENERATION_MODEL_VRAM.get(quantization, 14.0)
        
        # 현재 VRAM 여유 확인
        resolved_device = get_device(target_device_setting)
        free_vram = gpu_monitor.get_free_vram_gb(resolved_device)
        
        async def send_message(msg: str, msg_type: str = "system"):
            if session_id:
                await ws_manager.send_to_session(session_id, {"type": msg_type, "content": msg})
            else:
                await ws_manager.broadcast({"type": msg_type, "content": msg})
        
        # VRAM이 부족하면 편집 모델 언로드
        if not gpu_monitor.has_enough_vram(required_vram, resolved_device):
            if qwen_edit_manager.is_loaded:
                await send_message(f"⚠️ VRAM 부족 ({free_vram:.1f}GB < {required_vram:.1f}GB). 편집 모델을 언로드합니다...")
                await unload_edit_model_internal()
                
                # 언로드 후 VRAM 재확인
                await asyncio.sleep(0.5)  # GPU 캐시 정리 대기
                free_vram = gpu_monitor.get_free_vram_gb(resolved_device)
        
        # 여전히 부족하면 경고만 하고 진행 (CPU 오프로딩 가능)
        if not gpu_monitor.has_enough_vram(required_vram, resolved_device):
            await send_message(f"⚠️ VRAM이 여전히 부족합니다 ({free_vram:.1f}GB). CPU 오프로딩으로 시도합니다...")
            cpu_offload = True
        
        await send_message(f"🔄 생성 모델 자동 로드 중... ({quantization})")
        
        # 모델 로드 진행
        try:
            device = get_device(target_device_setting)
            
            quant_info = QUANTIZATION_OPTIONS.get(quantization)
            if not quant_info:
                return False, f"지원하지 않는 양자화: {quantization}"
            
            repo_id = quant_info["repo"]
            dtype = quant_info["type"]
            is_gguf = quant_info.get("is_gguf", False)
            
            # 진행 상황 브로드캐스트
            async def progress(percent, label, detail=""):
                await ws_manager.broadcast({
                    "type": "model_progress",
                    "progress": percent,
                    "label": label,
                    "detail": detail,
                    "stage": "loading" if percent < 100 else "complete"
                })
            
            await progress(5, "🔧 모델 자동 로드 시작...", f"양자화: {dtype}")
            
            from diffusers import ZImagePipeline
            
            if is_gguf:
                from diffusers import ZImageTransformer2DModel, GGUFQuantizationConfig
                from huggingface_hub import hf_hub_download
                
                filename = quant_info["filename"]
                
                await progress(15, "📥 GGUF 모델 다운로드 확인 중...", f"파일: {filename}")
                
                gguf_path = await asyncio.to_thread(
                    hf_hub_download,
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=str(MODELS_DIR)
                )
                
                await progress(35, "🔄 GGUF Transformer 로딩 중...", f"양자화 타입: {dtype}")
                
                transformer = await asyncio.to_thread(
                    ZImageTransformer2DModel.from_single_file,
                    gguf_path,
                    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                    torch_dtype=torch.bfloat16,
                )
                
                await progress(60, "🔗 파이프라인 구성 중...", "")
                
                pipe = await asyncio.to_thread(
                    ZImagePipeline.from_pretrained,
                    "Tongyi-MAI/Z-Image-Turbo",
                    transformer=transformer,
                    torch_dtype=torch.bfloat16,
                )
            else:
                await progress(15, "📥 모델 다운로드 확인 중...", f"저장소: {repo_id}")
                
                load_kwargs = {
                    "torch_dtype": torch.bfloat16,
                    "cache_dir": str(MODELS_DIR),
                }
                
                await progress(35, "🔄 모델 파일 로딩 중...", "")
                
                pipe = await asyncio.to_thread(
                    ZImagePipeline.from_pretrained,
                    repo_id,
                    **load_kwargs
                )
            
            await progress(80, f"🚀 {device.upper()}로 모델 전송 중...", "")
            
            if cpu_offload:
                await asyncio.to_thread(pipe.enable_model_cpu_offload)
            else:
                await asyncio.to_thread(pipe.to, device)
            
            current_model = quantization
            
            # GPU 모니터에 모델 등록
            gpu_monitor.register_model("Z-Image-Turbo", device)
            
            await progress(100, "✅ 모델 자동 로드 완료!", f"VRAM: {get_vram_info()}")
            
            await ws_manager.broadcast({
                "type": "model_status_change",
                "model_loaded": True,
                "current_model": current_model,
                "device": device
            })
            
            return True, f"생성 모델 자동 로드 완료: {quantization}"
            
        except Exception as e:
            await ws_manager.broadcast({
                "type": "model_progress",
                "progress": 0,
                "label": "❌ 자동 로드 실패",
                "detail": str(e),
                "stage": "error"
            })
            return False, f"생성 모델 자동 로드 실패: {str(e)}"


async def ensure_edit_model_loaded(session_id: str = None) -> tuple[bool, str]:
    """
    편집 모델이 로드되어 있는지 확인하고, 없으면 자동 로드
    VRAM이 부족하면 생성 모델을 먼저 언로드
    
    Args:
        session_id: 메시지를 보낼 세션 ID (None이면 broadcast)
    
    Returns:
        (success, message)
    """
    global pipe, current_model, edit_model_lock
    
    # 이미 로드되어 있으면 바로 반환
    if qwen_edit_manager.is_loaded:
        return True, "편집 모델이 이미 로드되어 있습니다."
    
    # 모델 잠금 확인
    if edit_model_lock.locked():
        return False, "다른 사용자가 편집 모델을 로드/언로드 중입니다. 잠시 후 다시 시도해주세요."
    
    async with edit_model_lock:
        # 다시 확인 (lock 대기 중 로드되었을 수 있음)
        if qwen_edit_manager.is_loaded:
            return True, "편집 모델이 이미 로드되어 있습니다."
        
        # 설정에서 옵션 가져오기 (Qwen은 4bit NF4 고정)
        cpu_offload = settings.get("edit_cpu_offload", True)
        target_device_setting = settings.get("edit_gpu", DEFAULT_GPU_SETTINGS["edit_gpu"])

        # 필요한 VRAM 계산 (Qwen-Image-Edit 4bit: ~16GB with CPU offload)
        required_vram = QWEN_EDIT_MODEL_VRAM
        
        # 현재 VRAM 여유 확인
        resolved_device = qwen_edit_manager.get_device(target_device_setting)
        free_vram = gpu_monitor.get_free_vram_gb(resolved_device)
        
        async def send_message(msg: str, msg_type: str = "edit_system"):
            if session_id:
                await ws_manager.send_to_session(session_id, {"type": msg_type, "content": msg})
            else:
                await ws_manager.broadcast({"type": msg_type, "content": msg})
        
        # VRAM이 부족하면 생성 모델 언로드
        if not gpu_monitor.has_enough_vram(required_vram, resolved_device):
            if pipe is not None:
                # 생성 모델 lock 확인
                if model_lock.locked():
                    return False, "생성 모델이 사용 중입니다. 잠시 후 다시 시도해주세요."
                
                async with model_lock:
                    await send_message(f"⚠️ VRAM 부족 ({free_vram:.1f}GB < {required_vram:.1f}GB). 생성 모델을 언로드합니다...")
                    await unload_generation_model_internal()
                
                # 언로드 후 VRAM 재확인
                await asyncio.sleep(0.5)  # GPU 캐시 정리 대기
                free_vram = gpu_monitor.get_free_vram_gb(resolved_device)
        
        # 여전히 부족하면 경고만 하고 진행 (CPU 오프로딩 활성화)
        if not gpu_monitor.has_enough_vram(required_vram, resolved_device):
            await send_message(f"⚠️ VRAM이 여전히 부족합니다 ({free_vram:.1f}GB). CPU 오프로딩으로 시도합니다...")
            cpu_offload = True
        
        await send_message("🔄 Qwen-Image-Edit 모델 자동 로드 중... (NF4 4bit)")

        # 진행 상황 콜백
        async def progress_callback(percent, label, detail):
            await ws_manager.broadcast({
                "type": "edit_model_progress",
                "progress": percent,
                "label": label,
                "detail": detail,
                "stage": "loading" if percent < 100 else "complete"
            })

        # 모델 로드
        success, message = await qwen_edit_manager.load_model(
            cpu_offload=cpu_offload,
            target_device=target_device_setting,
            progress_callback=progress_callback
        )
        
        if success:
            await ws_manager.broadcast({
                "type": "edit_model_status_change",
                "model_loaded": True,
                "current_model": qwen_edit_manager.current_model,
                "device": qwen_edit_manager.device
            })
        
        return success, message


# ============= 자동 언로드 관련 함수 =============
def update_activity():
    """마지막 활동 시간 업데이트"""
    global last_activity_time
    last_activity_time = time.time()


async def auto_unload_checker():
    """백그라운드에서 자동 언로드 체크"""
    global pipe, current_model, last_activity_time
    
    while True:
        await asyncio.sleep(60)  # 1분마다 체크
        
        # 자동 언로드 설정 확인
        if not settings.get("auto_unload_enabled", True):
            continue
        
        # 모델이 로드되어 있지 않으면 스킵
        if pipe is None:
            continue
        
        # 생성 중이면 스킵
        if generation_queue.is_processing():
            update_activity()  # 생성 중에는 활동으로 간주
            continue
        
        # 타임아웃 체크
        timeout_minutes = settings.get("auto_unload_timeout", 10)
        timeout_seconds = timeout_minutes * 60
        elapsed = time.time() - last_activity_time
        
        if elapsed >= timeout_seconds:
            print(f"⏰ 자동 언로드: {timeout_minutes}분 동안 활동이 없어 모델을 언로드합니다.")
            
            try:
                # GPU 모니터에서 모델 등록 해제
                gpu_monitor.unregister_model("Z-Image-Turbo")
                
                # 모델 언로드
                del pipe
                pipe = None
                current_model = None
                
                # GPU 캐시 정리 (생성 모델이 올려져 있던 디바이스만 정리)
                # - 편집 모델이 다른 GPU에 올라간 경우까지 영향을 주지 않도록 범위를 제한한다.
                gpu_monitor.clear_cache(device)
                gc.collect()
                
                # 클라이언트에게 알림
                await ws_manager.broadcast({
                    "type": "system",
                    "content": f"⏰ {timeout_minutes}분 동안 활동이 없어 모델이 자동 언로드되었습니다. VRAM을 절약합니다."
                })
                await ws_manager.broadcast({
                    "type": "model_progress", 
                    "progress": 100, 
                    "label": "⏰ 자동 언로드 완료",
                    "detail": f"VRAM 사용량: {get_vram_info()}",
                    "stage": "complete"
                })
                
                print(f"[OK] Auto-unload complete. VRAM: {get_vram_info()}")
                
            except Exception as e:
                print(f"[ERR] Auto-unload failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행되는 lifespan 핸들러"""
    global auto_unload_task
    
    # 시작 시: 자동 언로드 체크 태스크 시작
    auto_unload_task = asyncio.create_task(auto_unload_checker())
    print("[*] Auto unload checker started")
    
    # 큐 워커 시작
    await generation_queue.start_worker()
    print("[*] Image generation queue worker started")
    
    # 큐 콜백 설정
    generation_queue.set_callbacks(
        on_status_change=on_queue_status_change,
        on_broadcast=on_queue_broadcast,
        generate_func=execute_generation
    )
    
    yield
    
    # 종료 시: 태스크 취소
    if auto_unload_task:
        auto_unload_task.cancel()
        try:
            await auto_unload_task
        except asyncio.CancelledError:
            pass
    
    # 큐 워커 중지
    await generation_queue.stop_worker()


# ============= FastAPI 앱 설정 =============
app = FastAPI(
    title="Z-Image WebUI", 
    version="2.0.0", 
    lifespan=lifespan,
    swagger_ui_parameters={"persistAuthorization": True}  # 인증 정보 유지
)


# OpenAPI 스키마에 API 키 인증 추가
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    from fastapi.openapi.utils import get_openapi
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
    )
    
    # securitySchemes 추가
    openapi_schema["components"]["securitySchemes"] = {
        "APIKeyAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "API Key",
            "description": "API 키를 입력하세요 (zimg_로 시작하는 키). 설정 > API 키 관리에서 발급받을 수 있습니다."
        }
    }
    
    # API 키 인증이 필요한 엔드포인트에 security 설정 추가
    api_key_endpoints = [
        "/api/instant-generate",
        "/api/generate",
        "/api/edit/generate",
    ]
    
    for path_key, path_item in openapi_schema.get("paths", {}).items():
        if path_key in api_key_endpoints:
            for method in path_item.values():
                if isinstance(method, dict):
                    method["security"] = [{"APIKeyAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# 정적 파일 및 템플릿
app.mount("/static", StaticFiles(directory=ROOT_DIR / "static"), name="static")
templates = Jinja2Templates(directory=ROOT_DIR / "templates")


# ============= 라우터 등록 =============
from routers import auth, history, gallery, settings_router, admin

app.include_router(auth.router, prefix="/api/auth", tags=["인증"])
app.include_router(history.router, prefix="/api", tags=["히스토리"])
app.include_router(gallery.router, prefix="/api", tags=["갤러리"])
app.include_router(settings_router.router, prefix="/api", tags=["설정"])
app.include_router(admin.router, prefix="/api/admin", tags=["관리자"])


# ============= Pydantic 모델 =============
class GenerateRequest(BaseModel):
    prompt: str
    korean_prompt: str = ""  # 한국어 프롬프트 (원본)
    width: int = 512
    height: int = 512
    steps: int = 8
    guidance_scale: float = 0.0
    seed: int = -1
    num_images: int = 1
    auto_translate: bool = True


class ModelLoadRequest(BaseModel):
    quantization: str = "BF16 (기본, 최고품질)"
    cpu_offload: bool = False
    target_device: str = "auto"  # 관리자 전용: "auto", "cuda:0", "cuda:1", "cpu", "mps"


class SettingsRequest(BaseModel):
    openai_api_key: str = ""  # 레거시 호환
    output_path: str = ""
    filename_pattern: str = "{date}_{time}_{seed}"
    # LLM Provider 설정
    llm_provider: str = ""
    llm_api_key: str = ""
    # NOTE:
    # - /api/settings 는 다양한 설정(자동 언로드 등) 저장에도 재사용된다.
    # - 아래 값들을 기본값 ""로 두면, 요청 바디에 해당 필드가 없어도 Pydantic이 ""를 채워넣어
    #   저장 시 기존 값이 ""로 덮여서 "설정이 풀리는" 문제가 발생한다.
    # - 따라서 Optional로 두고, 실제로 값이 전달된 경우에만(= None이 아닐 때만) 저장한다.
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None
    # 시스템 프롬프트 (번역/향상)
    translate_system_prompt: Optional[str] = None
    enhance_system_prompt: Optional[str] = None
    # 자동 언로드 설정
    auto_unload_enabled: Optional[bool] = None
    auto_unload_timeout: Optional[int] = None
    # 편집 모델 자동 언로드 설정
    edit_auto_unload_enabled: Optional[bool] = None
    edit_auto_unload_timeout: Optional[int] = None

    # 모델 설정 (관리자 전용)
    quantization: Optional[str] = None
    cpu_offload: Optional[bool] = None
    # 편집 모델 설정 (관리자 전용) - Qwen은 4bit NF4 고정
    edit_cpu_offload: Optional[bool] = None


class FavoriteRequest(BaseModel):
    name: str
    prompt: str
    settings: dict = {}


class TranslateRequest(BaseModel):
    text: str


class EnhanceRequest(BaseModel):
    prompt: str
    style: str = "기본"


class ConversationUpdateRequest(BaseModel):
    conversation: List[Dict[str, Any]]


# ============= 편집 관련 Pydantic 모델 =============
class EditModelLoadRequest(BaseModel):
    model_path: str = ""
    cpu_offload: bool = True  # 기본 활성화 (VRAM 절약, ~16GB)
    target_device: str = "auto"  # 관리자 전용: "auto", "cuda:0", "cuda:1", "cpu", "mps"


class EditGenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = " "  # Qwen은 negative prompt 지원
    korean_prompt: str = ""
    steps: int = 20  # Qwen 기본값
    true_cfg_scale: float = 4.0  # Qwen 전용: 프롬프트 충실도
    guidance_scale: float = 1.0  # Qwen 기본값
    seed: int = -1
    num_images: int = 1
    auto_translate: bool = True


class EditTranslateRequest(BaseModel):
    text: str


class EditEnhanceRequest(BaseModel):
    instruction: str


class EditSuggestRequest(BaseModel):
    context: str = ""
    image_description: str = ""


class EditConversationUpdateRequest(BaseModel):
    conversation: List[Dict[str, Any]]


# ============= 인증 관련 Pydantic 모델 =============
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


class ResetPasswordRequest(BaseModel):
    new_password: Optional[str] = None  # None이면 임시 비밀번호 자동 생성


# ============= 유틸리티 함수 =============
def get_device(target_device: str = "auto") -> str:
    """
    사용할 디바이스 반환
    
    Args:
        target_device: 목표 디바이스 ("auto", "cuda:0", "cuda:1", "cpu", "mps")
    
    Returns:
        실제 사용할 디바이스
    """
    return gpu_monitor.resolve_device(target_device, prefer_empty=True)


def image_to_base64(image: Image.Image) -> str:
    """PIL 이미지를 base64로 변환"""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def get_vram_info() -> str:
    """VRAM 사용량 정보"""
    return gpu_monitor.get_vram_summary()


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


# ============= WebSocket 관리자는 services.ws_manager에서 가져옴 =============
# from services.ws_manager import ws_manager (상단에서 import)


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


# ============= 큐 콜백 함수들 =============
async def on_queue_status_change(session_id: str, event_type: str, data: dict):
    """큐 상태 변경 시 세션에 알림"""
    if event_type == "generation_start":
        await ws_manager.send_to_session(session_id, {
            "type": "queue_status",
            "status": "processing",
            "position": 0,
            "message": "🎨 이미지 생성을 시작합니다..."
        })
    elif event_type == "queue_position":
        await ws_manager.send_to_session(session_id, {
            "type": "queue_status",
            "status": "waiting",
            "position": data["position"],
            "message": f"⏳ 대기 중... (순서: {data['position']})"
        })
    elif event_type == "generation_error":
        await ws_manager.send_to_session(session_id, {
            "type": "error",
            "content": f"❌ 생성 오류: {data.get('error', '알 수 없는 오류')}"
        })
    elif event_type == "generation_complete":
        # 결과는 execute_generation에서 직접 전송
        pass


async def on_queue_broadcast(data: dict):
    """큐 상태 전체 브로드캐스트"""
    await ws_manager.broadcast(data)


async def execute_generation(request_data: dict) -> dict:
    """실제 이미지 생성 실행"""
    global pipe, current_model
    
    session_id = request_data.get("session_id")
    # 계정 단위(data_id)로 실행 시 SessionInfo가 없을 수 있으므로 옵션 처리
    session = session_manager.get_session(session_id)
    
    if pipe is None:
        raise Exception("모델이 로드되지 않았습니다.")
    
    prompt = request_data.get("prompt", "")
    korean_prompt = request_data.get("korean_prompt", "")
    width = request_data.get("width", 512)
    height = request_data.get("height", 512)
    steps = request_data.get("steps", 8)
    guidance_scale = request_data.get("guidance_scale", 0.0)
    seed = request_data.get("seed", -1)
    num_images = request_data.get("num_images", 1)
    auto_translate = request_data.get("auto_translate", True)
    
    # 번역
    final_prompt = prompt
    if auto_translate and translator.is_korean(prompt):
        await ws_manager.send_to_session(session_id, {
            "type": "system",
            "content": "🌐 프롬프트 번역 중..."
        })
        final_prompt, success = translator.translate(prompt)
        if not success:
            await ws_manager.send_to_session(session_id, {
                "type": "warning",
                "content": "⚠️ 번역 실패, 원문 사용"
            })
    
    # 시드 설정
    if seed == -1:
        seed = random.randint(0, 2147483647)
    
    # 생성 시작 메시지
    await ws_manager.send_to_session(session_id, {
        "type": "system",
        "content": "🎨 이미지 생성 중..."
    })
    
    # 생성 시작 전 GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    
    # 세션별 출력 디렉토리
    if session:
        outputs_dir = session.get_outputs_dir()
    else:
        # session_id가 user_{id} 형태면 outputs/user_{id}에 저장
        if isinstance(session_id, str) and session_id.startswith("user_"):
            outputs_dir = OUTPUTS_DIR / session_id
        else:
            outputs_dir = OUTPUTS_DIR
    
    images = []
    for i in range(num_images):
        current_seed = seed + i
        
        # 프로그레스 바 업데이트
        percent = ((i) / num_images) * 100
        await ws_manager.send_to_session(session_id, {
            "type": "image_progress",
            "progress": percent,
            "current": i + 1,
            "total": num_images
        })
        await asyncio.sleep(0.05)
        
        generator = torch.Generator(device).manual_seed(current_seed)
        loop = asyncio.get_running_loop()
        last_sent_step = {"value": -1}  # 클로저에서 mutable로 사용

        def _send_generation_progress_from_thread(current_step: int, total_steps: int):
            """diffusers 콜백(별도 스레드)에서 WebSocket 진행상황 전송"""
            # 너무 잦은 중복 전송 방지
            if current_step == last_sent_step["value"]:
                return
            last_sent_step["value"] = current_step

            # 전체 진행률 계산 (이미지 + 스텝 기준)
            image_progress = (i) / num_images
            step_progress = (current_step / max(total_steps, 1)) / num_images
            overall_progress = int((image_progress + step_progress) * 100)

            payload = {
                "type": "generation_progress",
                "current_image": i + 1,
                "total_images": num_images,
                "current_step": current_step,
                "total_steps": total_steps,
                "progress": overall_progress,
            }

            try:
                fut = asyncio.run_coroutine_threadsafe(
                    ws_manager.send_to_session(session_id, payload),
                    loop
                )
                # 예외가 발생해도 작업을 깨지 않도록 흡수
                fut.add_done_callback(lambda f: f.exception())
            except Exception:
                pass
        
        # 동기 pipe 호출을 스레드에서 실행
        def run_pipe():
            call_sig = inspect.signature(pipe.__call__)

            # diffusers 최신: callback_on_step_end 지원
            if "callback_on_step_end" in call_sig.parameters:
                def callback_on_step_end(_pipeline, step_index, _timestep, callback_kwargs):
                    # step_index는 0-based인 경우가 대부분
                    _send_generation_progress_from_thread(step_index + 1, steps)
                    return callback_kwargs

                return pipe(
                    prompt=final_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    callback_on_step_end=callback_on_step_end,
                ).images[0]

            # diffusers 구버전: callback/callback_steps 지원
            if "callback" in call_sig.parameters:
                def callback(step_index, _timestep, _latents):
                    _send_generation_progress_from_thread(step_index + 1, steps)

                return pipe(
                    prompt=final_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    callback=callback,
                    callback_steps=1,
                ).images[0]

            # 콜백 미지원(예외 케이스): 기존 동작
            return pipe(
                prompt=final_prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
        
        image = await asyncio.to_thread(run_pipe)
        
        # 메타데이터 생성 및 저장
        metadata = ImageMetadata.create_metadata(
            prompt=final_prompt,
            seed=current_seed,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            model=current_model or "unknown",
        )
        
        outputs_dir.mkdir(parents=True, exist_ok=True)
        filename = filename_generator.generate(
            pattern=settings.get("filename_pattern", "{date}_{time}_{seed}"),
            prompt=final_prompt,
            seed=current_seed
        )
        output_path = outputs_dir / filename
        ImageMetadata.save_with_metadata(image, output_path, metadata)
        
        images.append({
            "base64": image_to_base64(image),
            "filename": filename,
            "seed": current_seed,
            "path": (
                f"/outputs/{session.data_id}/{filename}"
                if session
                else (f"/outputs/{session_id}/{filename}" if isinstance(session_id, str) and session_id.startswith("user_") else f"/outputs/{filename}")
            )
        })
        
        # 각 이미지 생성 후 메모리 정리 (여러 장 생성 시 OOM 방지)
        if num_images > 1 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 생성 완료 후 GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    
    # 히스토리 추가 (사용자별)
    if session:
        history_mgr = get_history_manager_sync(session.data_id)
    else:
        # 계정 단위로 실행된 경우(user_{id})에는 그 계정 히스토리에 저장
        if isinstance(session_id, str) and session_id.startswith("user_"):
            history_mgr = get_history_manager_sync(session_id)
        else:
            from utils.history import history_manager
            history_mgr = history_manager
    
    history_entry = history_mgr.add(
        prompt=prompt,
        korean_prompt=korean_prompt,
        settings={
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
        }
    )
    
    # 완료 메시지
    await ws_manager.send_to_session(session_id, {
        "type": "complete",
        "content": f"✅ {len(images)}장 생성 완료! (시드: {seed})"
    })
    
    # 이미지 결과 전송
    await ws_manager.send_to_session(session_id, {
        "type": "generation_result",
        "images": images,
        "seed": seed,
        "prompt": final_prompt,
        "history_id": history_entry.id
    })
    
    return {
        "success": True,
        "images": images,
        "seed": seed,
        "prompt": final_prompt,
        "history_id": history_entry.id
    }


# ============= 세션별 출력 폴더 정적 파일 제공 =============
@app.get("/outputs/{session_id}/{filename:path}")
async def serve_session_output(session_id: str, filename: str, request: Request):
    """세션별 출력 파일 제공"""
    # 세션 ID 검증
    if not session_manager.validate_session_id(session_id):
        raise HTTPException(404, "파일을 찾을 수 없습니다.")
    
    file_path = OUTPUTS_DIR / session_id / filename
    if not file_path.exists():
        raise HTTPException(404, "파일을 찾을 수 없습니다.")
    
    return FileResponse(file_path)


# 레거시 출력 폴더 (세션 없는 기존 이미지용)
@app.get("/outputs/{filename:path}")
async def serve_legacy_output(filename: str):
    """레거시 출력 파일 제공"""
    # 세션 ID처럼 보이는지 확인 (UUID 형식)
    if "/" in filename or session_manager.validate_session_id(filename.split("/")[0] if "/" in filename else ""):
        raise HTTPException(404, "파일을 찾을 수 없습니다.")
    
    file_path = OUTPUTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(404, "파일을 찾을 수 없습니다.")
    
    return FileResponse(file_path)


# ============= API 엔드포인트 =============

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """메인 페이지 - 로그인 필수"""
    update_activity()
    session = await get_session_from_request(request)
    
    # 로그인하지 않은 경우 로그인 페이지로 리다이렉트
    if not session or not session.is_authenticated:
        from fastapi.responses import RedirectResponse
        response = RedirectResponse(url="/login", status_code=302)
        # 비로그인 세션은 생성/유지하지 않음 (쿠키도 제거)
        clear_session_cookie(response)
        return response
    
    response = templates.TemplateResponse("index.html", {
        "request": request,
        "user": {
            "id": session.user_id,
            "username": session.username,
        },
        # 정적 파일 캐시로 인해 UI 변경이 반영되지 않는 문제 방지
        "cache_bust": int(time.time()),
    })
    set_session_cookie(response, session)
    
    return response


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """로그인 페이지"""
    session = await get_session_from_request(request)
    
    # 이미 로그인된 경우 메인 페이지로 리다이렉트
    if session and session.is_authenticated:
        from fastapi.responses import RedirectResponse
        response = RedirectResponse(url="/", status_code=302)
        set_session_cookie(response, session)
        return response
    
    response = templates.TemplateResponse("login.html", {
        "request": request,
        "cache_bust": int(time.time()),
    })
    # 비로그인에서는 세션/쿠키를 만들지 않음
    clear_session_cookie(response)
    
    return response


@app.get("/api/status")
async def get_status(request: Request):
    """시스템 상태"""
    global pipe, current_model, device
    update_activity()
    
    session = await get_session_from_request(request)
    queue_status = generation_queue.get_queue_status()
    client_host = request.client.host if request.client else None
    is_admin = is_localhost(client_host)
    
    status = {
        "model_loaded": pipe is not None,
        "current_model": current_model,
        "device": device or get_device(),
        "vram": get_vram_info(),
        "is_generating": queue_status["is_processing"],
        "queue_length": queue_status["queue_length"],
        "connected_users": ws_manager.get_session_count(),
        # 프론트 호환 필드: 로그인 시 계정 키(user_{id}), 비로그인 시 None
        "session_id": (session.data_id if session else None),
        "is_admin": is_admin,
    }
    
    # 관리자인 경우 GPU 정보 추가
    if is_admin:
        status["gpu_info"] = {
            "gpu_count": gpu_monitor.gpu_count,
            "cuda_available": gpu_monitor.cuda_available,
            "available_devices": gpu_monitor.get_available_devices(),
            "gpus": gpu_monitor.get_all_gpu_info(),
        }
    
    return status


@app.post("/api/model/load")
async def load_model(request: Request, model_request: ModelLoadRequest):
    """모델 로드"""
    global pipe, current_model, device, model_lock
    
    # 모델 잠금 확인
    if model_lock.locked():
        raise HTTPException(409, "다른 사용자가 모델을 로드/언로드 중입니다. 잠시 후 다시 시도해주세요.")
    
    async with model_lock:
        # GPU 선택 (관리자만 특정 GPU 지정 가능)
        target_device = model_request.target_device
        client_host = request.client.host if request.client else None
        is_admin = is_localhost(client_host)

        # UI가 target_device="auto"로 보내는 경우가 많아서,
        # 관리자가 설정한 기본 GPU(설정 -> GPU 설정/모니터링)를 자동 적용한다.
        if target_device == "auto":
            target_device = settings.get("generation_gpu", DEFAULT_GPU_SETTINGS["generation_gpu"])

        if not is_admin and target_device != "auto":
            # 관리자가 아닌 경우 auto로 강제
            target_device = "auto"
        
        device = get_device(target_device)

        # 양자화/CPU 오프로딩은 관리자만 변경 가능
        requested_quantization = model_request.quantization
        requested_cpu_offload = model_request.cpu_offload
        if not is_admin:
            requested_quantization = settings.get("quantization", requested_quantization)
            requested_cpu_offload = settings.get("cpu_offload", requested_cpu_offload)

        quant_info = QUANTIZATION_OPTIONS.get(requested_quantization)
        
        if not quant_info:
            raise HTTPException(400, f"지원하지 않는 양자화: {requested_quantization}")
        
        repo_id = quant_info["repo"]
        dtype = quant_info["type"]
        is_gguf = quant_info.get("is_gguf", False)
        
        try:
            # 1단계: 로딩 준비
            await ws_manager.broadcast({
                "type": "model_progress", 
                "progress": 5, 
                "label": "🔧 모델 초기화 중...",
                "detail": f"양자화: {dtype}, 디바이스: {device}",
                "stage": "init"
            })
            await asyncio.sleep(0.1)
            
            from diffusers import ZImagePipeline
            
            if is_gguf:
                # GGUF 양자화 모델 로드
                from diffusers import ZImageTransformer2DModel, GGUFQuantizationConfig
                from huggingface_hub import hf_hub_download
                
                filename = quant_info["filename"]
                
                # 2단계: GGUF 다운로드
                await ws_manager.broadcast({
                    "type": "model_progress", 
                    "progress": 10, 
                    "label": "📥 GGUF 모델 다운로드 확인 중...",
                    "detail": f"파일: {filename} (캐시 확인 중...)",
                    "stage": "download"
                })
                await asyncio.sleep(0.1)
                
                gguf_path = await asyncio.to_thread(
                    hf_hub_download,
                    repo_id=repo_id, 
                    filename=filename,
                    cache_dir=str(MODELS_DIR)
                )
                
                # 3단계: GGUF Transformer 로드
                await ws_manager.broadcast({
                    "type": "model_progress", 
                    "progress": 30, 
                    "label": "🔄 GGUF Transformer 로딩 중...",
                    "detail": f"양자화 타입: {dtype} (시간이 걸릴 수 있습니다)",
                    "stage": "load_transformer"
                })
                await asyncio.sleep(0.1)
                
                transformer = await asyncio.to_thread(
                    ZImageTransformer2DModel.from_single_file,
                    gguf_path,
                    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                    torch_dtype=torch.bfloat16,
                )
                
                # 4단계: 파이프라인 구성
                await ws_manager.broadcast({
                    "type": "model_progress", 
                    "progress": 55, 
                    "label": "🔗 파이프라인 구성 중...",
                    "detail": "기본 모델 다운로드/로드 및 GGUF Transformer 결합",
                    "stage": "load_pipeline"
                })
                await asyncio.sleep(0.1)
                
                pipe = await asyncio.to_thread(
                    ZImagePipeline.from_pretrained,
                    "Tongyi-MAI/Z-Image-Turbo",
                    transformer=transformer,
                    torch_dtype=torch.bfloat16,
                )
            else:
                # 기본 BF16 모델 로드
                await ws_manager.broadcast({
                    "type": "model_progress", 
                    "progress": 15, 
                    "label": "📥 모델 다운로드 확인 중...",
                    "detail": f"저장소: {repo_id} (캐시에 없으면 다운로드합니다)",
                    "stage": "download"
                })
                await asyncio.sleep(0.1)
                
                load_kwargs = {
                    "torch_dtype": torch.bfloat16,
                    "cache_dir": str(MODELS_DIR),
                }
                
                await ws_manager.broadcast({
                    "type": "model_progress", 
                    "progress": 30, 
                    "label": "🔄 모델 파일 로딩 중...",
                    "detail": "다운로드 또는 캐시에서 로드 중... (처음 실행 시 몇 분 소요)",
                    "stage": "load_model"
                })
                await asyncio.sleep(0.1)
                
                pipe = await asyncio.to_thread(
                    ZImagePipeline.from_pretrained,
                    repo_id,
                    **load_kwargs
                )
            
            # 5단계: 디바이스 전송
            await ws_manager.broadcast({
                "type": "model_progress", 
                "progress": 75, 
                "label": f"🚀 {device.upper()}로 모델 전송 중...",
                "detail": "VRAM으로 모델 복사 중...",
                "stage": "to_device"
            })
            await asyncio.sleep(0.1)
            
            if requested_cpu_offload:
                await asyncio.to_thread(pipe.enable_model_cpu_offload)
                await ws_manager.broadcast({
                    "type": "model_progress", 
                    "progress": 95, 
                    "label": "⚙️ CPU 오프로딩 설정 중...",
                    "detail": "VRAM 부족 시 자동으로 RAM 사용",
                    "stage": "cpu_offload"
                })
            else:
                await asyncio.to_thread(pipe.to, device)
            
            current_model = requested_quantization
            
            # GPU 모니터에 모델 등록
            gpu_monitor.register_model("Z-Image-Turbo", device)
            
            # 6단계: 완료
            await ws_manager.broadcast({
                "type": "model_progress", 
                "progress": 100, 
                "label": "✅ 모델 로드 완료!",
                "detail": f"VRAM 사용량: {get_vram_info()}",
                "stage": "complete"
            })
            
            await ws_manager.broadcast({
                "type": "model_status_change",
                "model_loaded": True,
                "current_model": current_model,
                "device": device
            })
            
            await ws_manager.broadcast({
                "type": "complete",
                "content": f"✅ 모델 로드 완료! ({dtype}, {device})"
            })
            
            return {"success": True, "message": f"모델 로드 완료: {repo_id} ({dtype})", "device": device}
            
        except Exception as e:
            await ws_manager.broadcast({
                "type": "model_progress", 
                "progress": 0, 
                "label": "❌ 로드 실패",
                "detail": str(e),
                "stage": "error"
            })
            await ws_manager.broadcast({"type": "error", "content": f"❌ 모델 로드 실패: {str(e)}"})
            raise HTTPException(500, str(e))


@app.post("/api/model/unload")
async def unload_model(request: Request):
    """모델 언로드"""
    global pipe, current_model, model_lock
    
    # 모델 잠금 확인
    if model_lock.locked():
        raise HTTPException(409, "다른 사용자가 모델을 로드/언로드 중입니다.")
    
    async with model_lock:
        if pipe is None:
            return {"success": True, "message": "로드된 모델이 없습니다."}
        
        try:
            await ws_manager.broadcast({
                "type": "model_progress", 
                "progress": 30, 
                "label": "모델 메모리 해제 중...",
                "detail": ""
            })
            
            # GPU 모니터에서 모델 등록 해제
            gpu_monitor.unregister_model("Z-Image-Turbo")
            
            del pipe
            pipe = None
            current_model = None
            
            await ws_manager.broadcast({
                "type": "model_progress", 
                "progress": 60, 
                "label": "VRAM 정리 중...",
                "detail": ""
            })
            
            # GPU 캐시 정리 (생성 모델 디바이스만 정리)
            gpu_monitor.clear_cache(device)
            gc.collect()
            
            await ws_manager.broadcast({
                "type": "model_progress", 
                "progress": 100, 
                "label": "언로드 완료!",
                "detail": f"VRAM 사용량: {get_vram_info()}"
            })
            
            await ws_manager.broadcast({
                "type": "model_status_change",
                "model_loaded": False,
                "current_model": None
            })
            
            await ws_manager.broadcast({"type": "complete", "content": "✅ 모델 언로드 완료!"})
            return {"success": True, "message": "모델 언로드 완료"}
            
        except Exception as e:
            raise HTTPException(500, str(e))


# ============= Instant Generate API (휘발성 이미지 생성) =============
class InstantGenerateRequest(BaseModel):
    """휘발성 이미지 생성 요청"""
    prompt: str
    width: int = 512
    height: int = 512
    steps: int = 8
    guidance_scale: float = 0.0
    seed: int = -1
    num_images: int = 1
    auto_translate: bool = True


@app.post("/api/instant-generate", summary="Instant Generate (No Save)", 
          description="휘발성 이미지 생성 - 파일 저장 없이 메모리에서 바로 반환 (API 키 필수)")
async def instant_generate_image(
    request: Request,
    gen_request: InstantGenerateRequest,
    api_key: Optional[str] = Depends(get_api_key_auth)
):
    """
    휘발성 이미지 생성 API
    
    - 이미지를 파일로 저장하지 않음
    - 히스토리에 기록하지 않음
    - base64로 직접 반환 후 메모리에서 삭제
    - API 키 인증 필수
    """
    global pipe
    update_activity()
    
    # API 키 인증 필수
    api_key_str = api_key or get_api_key_from_request(request)
    if not api_key_str:
        raise HTTPException(401, "API 키가 필요합니다. Authorization: Bearer <api_key> 헤더를 사용하세요.")
    
    is_valid, api_key_obj = api_key_manager.validate_api_key(api_key_str)
    if not is_valid:
        raise HTTPException(401, "유효하지 않은 API 키입니다.")
    
    # 모델 체크
    if pipe is None:
        success, message = await ensure_generation_model_loaded()
        if not success:
            raise HTTPException(400, f"모델 자동 로드 실패: {message}")
    
    if not gen_request.prompt.strip():
        raise HTTPException(400, "프롬프트를 입력해주세요.")
    
    try:
        # 프롬프트 번역
        final_prompt = gen_request.prompt
        if gen_request.auto_translate and translator.is_korean(gen_request.prompt):
            final_prompt, success = translator.translate(gen_request.prompt)
        
        # 시드 설정
        seed = gen_request.seed
        if seed == -1:
            seed = random.randint(0, 2147483647)
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        # 이미지 생성
        images_response = []
        current_seed = seed
        
        for i in range(gen_request.num_images):
            generator = torch.Generator(device=device).manual_seed(current_seed)
            
            result = pipe(
                prompt=final_prompt,
                width=gen_request.width,
                height=gen_request.height,
                num_inference_steps=gen_request.steps,
                guidance_scale=gen_request.guidance_scale,
                generator=generator,
            )
            
            result_image = result.images[0]
            
            # base64로 변환 (파일 저장 없음)
            buffered = BytesIO()
            result_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            buffered.close()
            
            images_response.append({
                "base64": img_base64,
                "seed": current_seed,
                "width": result_image.width,
                "height": result_image.height,
            })
            
            # 다음 이미지를 위한 시드 증가
            current_seed += 1
            
            # 메모리 정리
            del result_image
            del result
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return {
            "success": True,
            "images": images_response,
            "prompt": final_prompt,
            "original_prompt": gen_request.prompt,
            "settings": {
                "width": gen_request.width,
                "height": gen_request.height,
                "steps": gen_request.steps,
                "guidance_scale": gen_request.guidance_scale,
                "seed": seed,
                "num_images": gen_request.num_images,
            }
        }
        
    except Exception as e:
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        raise HTTPException(500, f"이미지 생성 실패: {str(e)}")


@app.post("/api/generate", summary="Generate Image", description="이미지 생성 요청 (큐에 추가 또는 직접 실행)")
async def generate_image(
    request: Request, 
    gen_request: GenerateRequest,
    api_key: Optional[str] = Depends(get_api_key_auth)
):
    """이미지 생성 요청 (큐에 추가 또는 직접 실행)"""
    update_activity()
    
    # API 키 또는 세션 인증 확인 (Swagger docs의 Authorize 버튼 또는 헤더에서)
    api_key_str = api_key or get_api_key_from_request(request)
    
    if api_key_str:
        # API 키 인증
        is_valid, api_key_obj = api_key_manager.validate_api_key(api_key_str)
        if not is_valid:
            raise HTTPException(401, "유효하지 않은 API 키입니다.")
        
        # API 키로 호출 시: 직접 실행 모드 (동기적으로 결과 반환)
        if pipe is None:
            success, message = await ensure_generation_model_loaded()
            if not success:
                raise HTTPException(400, f"모델 자동 로드 실패: {message}")
        
        if not gen_request.prompt.strip():
            raise HTTPException(400, "프롬프트를 입력해주세요.")
        
        # API 키 사용 시 별도 data_id 사용
        api_data_id = f"api_key_{api_key_obj.id}"
        
        # 직접 이미지 생성 실행 (큐 없이)
        try:
            request_data = {
                "session_id": api_data_id,
                "prompt": gen_request.prompt,
                "korean_prompt": gen_request.korean_prompt,
                "width": gen_request.width,
                "height": gen_request.height,
                "steps": gen_request.steps,
                "guidance_scale": gen_request.guidance_scale,
                "seed": gen_request.seed,
                "num_images": gen_request.num_images,
                "auto_translate": gen_request.auto_translate,
            }
            result = await execute_generation(request_data)
            return result
        except Exception as e:
            raise HTTPException(500, f"이미지 생성 실패: {str(e)}")
    
    # 기존 세션 인증 방식
    session = await get_session_from_request(request)
    require_auth(session)
    
    # 모델이 로드되지 않았으면 자동 로드
    if pipe is None:
        success, message = await ensure_generation_model_loaded(session.data_id)
        if not success:
            raise HTTPException(400, f"모델 자동 로드 실패: {message}")
    
    if not gen_request.prompt.strip():
        raise HTTPException(400, "프롬프트를 입력해주세요.")
    
    # Rate limit 체크
    # 계정 단위로 제한(세션 단위 구분 제거)
    exceeded, count = session_manager.check_rate_limit(session.session_id)
    if exceeded:
        await ws_manager.send_to_session(session.data_id, {
            "type": "warning",
            "content": f"⚠️ 요청이 너무 많습니다. (분당 {count}회) 잠시 후 다시 시도해주세요."
        })
    
    # 요청 데이터 준비
    request_data = {
        # 실행/알림/큐 모두 계정(data_id) 단위로 처리
        "session_id": session.data_id,
        "prompt": gen_request.prompt,
        "korean_prompt": gen_request.korean_prompt,
        "width": gen_request.width,
        "height": gen_request.height,
        "steps": gen_request.steps,
        "guidance_scale": gen_request.guidance_scale,
        "seed": gen_request.seed,
        "num_images": gen_request.num_images,
        "auto_translate": gen_request.auto_translate,
    }
    
    # 큐에 추가
    item_id, position = await generation_queue.add_to_queue(session.data_id, request_data)
    
    # 큐 상태 알림
    if position > 1:
        await ws_manager.send_to_session(session.data_id, {
            "type": "queue_status",
            "status": "queued",
            "position": position,
            "message": f"⏳ 대기열에 추가되었습니다. (순서: {position})"
        })
    else:
        await ws_manager.send_to_session(session.data_id, {
            "type": "queue_status",
            "status": "processing",
            "position": 0,
            "message": "🎨 이미지 생성을 시작합니다..."
        })
    
    response = JSONResponse(content={
        "success": True,
        "queued": True,
        "item_id": item_id,
        "position": position,
        "message": f"요청이 큐에 추가되었습니다. (순서: {position})"
    })
    set_session_cookie(response, session)
    
    return response


@app.post("/api/preview")
async def generate_preview(
    request: Request, 
    gen_request: GenerateRequest,
    api_key: Optional[str] = Depends(get_api_key_auth)
):
    """빠른 미리보기 (256x256)"""
    gen_request.width = 256
    gen_request.height = 256
    gen_request.steps = min(gen_request.steps, 4)
    gen_request.num_images = 1
    return await generate_image(request, gen_request, api_key)


@app.post("/api/translate")
async def translate_text(request: Request, trans_request: TranslateRequest):
    """프롬프트 번역 (한국어 → 영어)"""
    update_activity()
    from utils.llm_client import llm_client
    
    if not llm_client.is_available:
        raise HTTPException(400, "LLM API가 설정되지 않았습니다.")
    
    translated, success = translator.translate(trans_request.text)
    return {"success": success, "translated": translated}


@app.post("/api/translate-reverse")
async def reverse_translate_text(request: Request, trans_request: TranslateRequest):
    """프롬프트 역번역 (영어 → 한국어)"""
    update_activity()
    from utils.llm_client import llm_client
    
    if not llm_client.is_available:
        raise HTTPException(400, "LLM API가 설정되지 않았습니다.")
    
    translated, success = translator.reverse_translate(trans_request.text)
    return {"success": success, "translated": translated}


@app.post("/api/enhance")
async def enhance_prompt(request: Request, enhance_request: EnhanceRequest):
    """프롬프트 향상"""
    update_activity()
    from utils.llm_client import llm_client
    
    if not llm_client.is_available:
        raise HTTPException(400, "LLM API가 설정되지 않았습니다.")
    
    enhanced, success = prompt_enhancer.enhance(enhance_request.prompt, enhance_request.style)
    return {"success": success, "enhanced": enhanced}


@app.get("/api/templates")
async def get_templates():
    """프롬프트 템플릿 목록"""
    return {"templates": PROMPT_TEMPLATES}


@app.get("/api/model-status")
async def get_model_download_status():
    """각 모델의 다운로드 상태 확인"""
    from huggingface_hub import try_to_load_from_cache
    
    status = {}
    
    for option_name, option_info in QUANTIZATION_OPTIONS.items():
        is_downloaded = False
        
        try:
            if option_info.get("is_gguf", False):
                filename = option_info.get("filename", "")
                repo_id = option_info.get("repo", "")
                
                if filename and repo_id:
                    cached_path = try_to_load_from_cache(
                        repo_id=repo_id,
                        filename=filename
                    )
                    is_downloaded = cached_path is not None
            else:
                repo_id = option_info.get("repo", "")
                if repo_id:
                    cached_path = try_to_load_from_cache(
                        repo_id=repo_id,
                        filename="model_index.json"
                    )
                    is_downloaded = cached_path is not None
        except Exception as e:
            print(f"캐시 확인 오류 ({option_name}): {e}")
            is_downloaded = False
        
        status[option_name] = is_downloaded
    
    return {"status": status}


# ============= 히스토리/즐겨찾기/갤러리 API는 routers/history.py, routers/gallery.py로 이동됨 =============


# ============= 설정 API는 routers/settings_router.py로 이동됨 =============


# ============= 인증 API는 routers/auth.py로 이동됨 =============


# ============= 관리자 API는 routers/admin.py로 이동됨 =============


# ============= WebSocket =============
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, z_image_session: Optional[str] = Cookie(default=None)):
    """웹소켓 연결 (세션별)"""
    # 세션 가져오기
    # 비로그인에서 세션을 만들지 않기 위해 "기존 세션 조회"만 시도
    session = session_manager.get_session(z_image_session) if z_image_session else None
    
    # 로그인하지 않은 경우 연결 거부
    if not session or not session.is_authenticated:
        await websocket.close(code=4001, reason="로그인이 필요합니다.")
        return
    
    # 계정(data_id) 단위로 WebSocket 룸을 통일 (세션 구분 제거)
    await ws_manager.connect(websocket, session.data_id)
    update_activity()
    
    try:
        # 연결 시 상태 전송
        await websocket.send_json({
            "type": "connected",
            "content": "서버에 연결되었습니다.",
            # 프론트에서 쓰는 값도 계정 키로 통일
            "session_id": session.data_id,
            "connected_users": ws_manager.get_session_count()
        })
        
        # 현재 모델 상태 전송
        await websocket.send_json({
            "type": "model_status_change",
            "model_loaded": pipe is not None,
            "current_model": current_model
        })
        
        # 편집 모델 상태 전송
        await websocket.send_json({
            "type": "edit_model_status_change",
            "model_loaded": qwen_edit_manager.is_loaded,
            "current_model": qwen_edit_manager.current_model
        })
        
        # 접속자 수 브로드캐스트
        await ws_manager.broadcast({
            "type": "user_count",
            "count": ws_manager.get_session_count()
        })
        
        while True:
            data = await websocket.receive_text()
            update_activity()
            
            # 클라이언트 메시지 처리
            try:
                message = json.loads(data)
                # 핑/퐁 처리
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except:
                pass
            
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
        
        # 접속자 수 브로드캐스트
        await ws_manager.broadcast({
            "type": "user_count",
            "count": ws_manager.get_session_count()
        })


# ============= Qwen-Image-Edit API =============

def update_edit_activity():
    """편집 모델 마지막 활동 시간 업데이트"""
    global edit_last_activity_time
    edit_last_activity_time = time.time()


@app.get("/api/edit/status")
async def get_edit_status(request: Request):
    """편집 모델 상태"""
    update_edit_activity()
    
    session = await get_session_from_request(request)
    require_auth(session)
    client_host = request.client.host if request.client else None
    is_admin = is_localhost(client_host)
    
    status = {
        "model_loaded": qwen_edit_manager.is_loaded,
        "current_model": qwen_edit_manager.current_model,
        "current_quantization": "NF4 (4bit)",  # Qwen은 4bit NF4 고정
        "cpu_offload_enabled": qwen_edit_manager.cpu_offload_enabled,
        # 저장된(기본) 편집 모델 설정값 - 새로고침/재시작 후 UI에서 유지되도록 제공
        "saved_edit_cpu_offload": settings.get("edit_cpu_offload", True),
        "device": qwen_edit_manager.device or qwen_edit_manager.get_device(),
        "vram": get_vram_info(),
        "session_id": session.data_id,
        "is_admin": is_admin,
        # Qwen은 4bit NF4 고정 (~16GB with CPU offload)
        "quantization_options": ["NF4 (4bit)"],
        "quantization_details": {
            "NF4 (4bit)": {
                "type": "nf4",
                "estimated_vram": "~16GB (CPU offload)",
            }
        },
    }
    
    # 관리자인 경우 추가 정보
    if is_admin:
        status["available_devices"] = gpu_monitor.get_available_devices()
    
    return status


@app.post("/api/edit/model/load")
async def load_edit_model(request: Request, model_request: EditModelLoadRequest):
    """Qwen-Image-Edit 모델 로드"""
    global edit_model_lock
    
    if edit_model_lock.locked():
        raise HTTPException(409, "다른 사용자가 모델을 로드/언로드 중입니다.")
    
    # GPU 선택 (관리자만 특정 GPU 지정 가능)
    target_device = model_request.target_device
    client_host = request.client.host if request.client else None
    is_admin = is_localhost(client_host)

    # UI가 target_device="auto"로 보내는 경우가 많아서,
    # 관리자가 설정한 편집 기본 GPU를 자동 적용한다.
    if target_device == "auto":
        target_device = settings.get("edit_gpu", DEFAULT_GPU_SETTINGS["edit_gpu"])

    if not is_admin and target_device != "auto":
        # 관리자가 아닌 경우 auto로 강제
        target_device = "auto"
    
    async with edit_model_lock:
        async def progress_callback(percent, label, detail):
            await ws_manager.broadcast({
                "type": "edit_model_progress",
                "progress": percent,
                "label": label,
                "detail": detail,
                "stage": "loading" if percent < 100 else "complete"
            })
        
        try:
            await ws_manager.broadcast({
                "type": "edit_model_progress",
                "progress": 0,
                "label": "🔧 편집 모델 로드 시작...",
                "detail": "",
                "stage": "init"
            })
            
            # CPU 오프로딩은 관리자만 변경 가능 (Qwen은 4bit NF4 고정)
            requested_cpu_offload = model_request.cpu_offload
            if not is_admin:
                requested_cpu_offload = settings.get("edit_cpu_offload", requested_cpu_offload)

            success, message = await qwen_edit_manager.load_model(
                cpu_offload=requested_cpu_offload,
                model_path=model_request.model_path if model_request.model_path else None,
                target_device=target_device,
                progress_callback=progress_callback
            )
            
            if success:
                await ws_manager.broadcast({
                    "type": "edit_model_status_change",
                    "model_loaded": True,
                    "current_model": qwen_edit_manager.current_model,
                    "device": qwen_edit_manager.device
                })
                await ws_manager.broadcast({
                    "type": "edit_system",
                    "content": f"✅ 편집 모델 로드 완료! ({qwen_edit_manager.device})"
                })
                return {"success": True, "message": message, "device": qwen_edit_manager.device}
            else:
                await ws_manager.broadcast({
                    "type": "edit_model_progress",
                    "progress": 0,
                    "label": "❌ 로드 실패",
                    "detail": message,
                    "stage": "error"
                })
                raise HTTPException(500, message)
                
        except Exception as e:
            await ws_manager.broadcast({
                "type": "edit_system",
                "content": f"❌ 편집 모델 로드 실패: {str(e)}"
            })
            raise HTTPException(500, str(e))


@app.post("/api/edit/model/unload")
async def unload_edit_model(request: Request):
    """Qwen-Image-Edit 모델 언로드"""
    global edit_model_lock
    
    if edit_model_lock.locked():
        raise HTTPException(409, "다른 사용자가 모델을 로드/언로드 중입니다.")
    
    async with edit_model_lock:
        try:
            await ws_manager.broadcast({
                "type": "edit_model_progress",
                "progress": 50,
                "label": "편집 모델 언로드 중...",
                "detail": ""
            })
            
            success, message = await qwen_edit_manager.unload_model()
            
            await ws_manager.broadcast({
                "type": "edit_model_progress",
                "progress": 100,
                "label": "언로드 완료!",
                "detail": f"VRAM: {get_vram_info()}"
            })
            
            await ws_manager.broadcast({
                "type": "edit_model_status_change",
                "model_loaded": False,
                "current_model": None
            })
            
            await ws_manager.broadcast({
                "type": "complete",
                "content": "✅ 편집 모델 언로드 완료!"
            })
            
            return {"success": success, "message": message}
            
        except Exception as e:
            raise HTTPException(500, str(e))


@app.post("/api/edit/generate", summary="Edit Image", description="이미지 편집 실행 (Qwen)")
async def edit_image(
    request: Request,
    images: List[UploadFile] = File(..., description="편집할 이미지 (1~3장)"),
    prompt: str = Form(...),
    negative_prompt: str = Form(" "),
    korean_prompt: str = Form(""),
    steps: int = Form(20),
    true_cfg_scale: float = Form(4.0),
    guidance_scale: float = Form(1.0),
    seed: int = Form(-1),
    num_images: int = Form(1),
    auto_translate: str = Form("true"),
    api_key: Optional[str] = Depends(get_api_key_auth)
):
    """이미지 편집 실행 (Qwen - 1~3장 이미지 입력 지원)"""
    update_edit_activity()
    
    # API 키 또는 세션 인증 확인 (Swagger docs의 Authorize 버튼 또는 헤더에서)
    api_key_str = api_key or get_api_key_from_request(request)
    api_key_obj = None
    session = None
    data_id = None
    use_websocket = True
    
    if api_key_str:
        # API 키 인증
        is_valid, api_key_obj = api_key_manager.validate_api_key(api_key_str)
        if not is_valid:
            raise HTTPException(401, "유효하지 않은 API 키입니다.")
        data_id = f"api_key_{api_key_obj.id}"
        use_websocket = False  # API 키 사용 시 웹소켓 알림 비활성화
    else:
        # 기존 세션 인증
        session = await get_session_from_request(request)
        require_auth(session)
        data_id = session.data_id
    
    # 편집 모델이 로드되지 않았으면 자동 로드
    if not qwen_edit_manager.is_loaded:
        success, message = await ensure_edit_model_loaded(data_id if use_websocket else None)
        if not success:
            raise HTTPException(400, f"편집 모델 자동 로드 실패: {message}")
    
    if not prompt.strip():
        raise HTTPException(400, "편집 프롬프트를 입력해주세요.")
    
    # Form에서 받은 auto_translate 문자열을 bool로 변환
    auto_translate_bool = auto_translate.lower() in ("true", "1", "yes")
    
    try:
        # 이번 편집 요청의 고유 ID (입력/참조 이미지 파일명 등에 사용)
        run_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        
        # 출력 디렉토리 (세션 또는 API 키별)
        if session:
            outputs_dir = session.get_outputs_dir()
        else:
            outputs_dir = OUTPUTS_DIR / data_id
        outputs_dir.mkdir(parents=True, exist_ok=True)
        
        # 이미지 로드 (1~3장)
        if len(images) > 3:
            raise HTTPException(400, "최대 3장의 이미지만 업로드할 수 있습니다.")
        
        pil_images = []
        original_image_urls = []
        
        for idx, img_file in enumerate(images):
            image_data = await img_file.read()
            pil_image = Image.open(BytesIO(image_data)).convert("RGB")
            pil_images.append(pil_image)
            
            # 업로드된 원본 이미지를 출력 폴더에 저장 (편집기록에서 원본 확인용)
            original_filename = f"edit_input_{run_id}_{idx+1}.png"
            original_output_path = outputs_dir / original_filename
            pil_image.save(original_output_path, format="PNG")
            original_image_urls.append(f"/outputs/{data_id}/{original_filename}")
        
        # 프롬프트 번역
        final_prompt = prompt
        if auto_translate_bool and edit_translator.is_korean(prompt):
            if use_websocket:
                await ws_manager.send_to_session(data_id, {
                    "type": "edit_system",
                    "content": "🌐 편집 지시어 번역 중..."
                })
            final_prompt, success = edit_translator.translate(prompt)
            if not success and use_websocket:
                await ws_manager.send_to_session(data_id, {
                    "type": "edit_system",
                    "content": "⚠️ 번역 실패, 원문 사용"
                })
        
        # 편집 시작 메시지
        if use_websocket:
            await ws_manager.send_to_session(data_id, {
                "type": "edit_system",
                "content": "🎨 이미지 편집 중..."
            })
        
        # 진행 상황 콜백 정의
        async def edit_progress_callback(current_image: int, total_images: int, current_step: int, total_steps: int):
            if not use_websocket:
                return
            # 전체 진행률 계산 (이미지 + 스텝 기준)
            image_progress = (current_image - 1) / total_images
            step_progress = current_step / total_steps / total_images
            overall_progress = int((image_progress + step_progress) * 100)
            
            await ws_manager.send_to_session(data_id, {
                "type": "edit_progress",
                "current_image": current_image,
                "total_images": total_images,
                "current_step": current_step,
                "total_steps": total_steps,
                "progress": overall_progress
            })
        
        # 상태 메시지 콜백 정의 (참조 이미지 분석 등)
        async def edit_status_callback(message: str):
            if not use_websocket:
                return
            await ws_manager.send_to_session(data_id, {
                "type": "edit_system",
                "content": message
            })
        
        # 편집 실행 (Qwen)
        success, results, message = await qwen_edit_manager.edit_image(
            images=pil_images,
            prompt=final_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            true_cfg_scale=true_cfg_scale,
            guidance_scale=guidance_scale,
            seed=seed,
            num_images=num_images,
            progress_callback=edit_progress_callback,
            status_callback=edit_status_callback
        )
        
        if not success:
            raise HTTPException(500, message)
        
        # 결과 저장 및 반환
        images_response = []
        result_paths = []
        
        for i, result in enumerate(results):
            result_image = result["image"]
            seed = result["seed"]
            
            # 파일명 생성
            filename = filename_generator.generate(
                pattern=settings.get("filename_pattern", "{date}_{time}_{seed}"),
                prompt=final_prompt,
                seed=seed
            )
            filename = f"edit_{filename}"
            output_path = outputs_dir / filename
            
            # 메타데이터와 함께 저장
            metadata = ImageMetadata.create_metadata(
                prompt=final_prompt,
                seed=seed,
                width=result_image.width,
                height=result_image.height,
                steps=steps,
                guidance_scale=guidance_scale,
                model="Qwen-Image-Edit",
            )
            ImageMetadata.save_with_metadata(result_image, output_path, metadata)
            
            result_paths.append(str(output_path))
            images_response.append({
                "base64": image_to_base64(result_image),
                "filename": filename,
                "seed": seed,
                "path": f"/outputs/{data_id}/{filename}"
            })
        
        # 히스토리 저장
        edit_history_mgr = get_edit_history_manager_sync(data_id)
        history_entry = edit_history_mgr.add(
            prompt=final_prompt,
            negative_prompt=negative_prompt,
            korean_prompt=korean_prompt,
            settings={
                "steps": steps,
                "true_cfg_scale": true_cfg_scale,
                "guidance_scale": guidance_scale,
                "seed": results[0]["seed"] if results else -1,
            },
            original_image_paths=original_image_urls,
            result_image_paths=[img["path"] for img in images_response]
        )
        
        # 완료 메시지
        if use_websocket:
            await ws_manager.send_to_session(data_id, {
                "type": "edit_system",
                "content": f"✅ 편집 완료! (시드: {results[0]['seed'] if results else 'N/A'})"
            })
            
            # 결과 전송
            await ws_manager.send_to_session(data_id, {
                "type": "edit_result",
                "images": images_response,
                "seed": results[0]["seed"] if results else -1,
                "prompt": final_prompt,
                "history_id": history_entry.id
            })
        
        return {
            "success": True,
            "images": images_response,
            "seed": results[0]["seed"] if results else -1,
            "prompt": final_prompt,
            "history_id": history_entry.id
        }
        
    except Exception as e:
        if use_websocket:
            await ws_manager.send_to_session(data_id, {
                "type": "error",
                "content": f"❌ 편집 오류: {str(e)}"
            })
        raise HTTPException(500, str(e))


@app.post("/api/edit/translate")
async def translate_edit_instruction(request: Request, trans_request: EditTranslateRequest):
    """편집 지시어 번역"""
    update_edit_activity()
    from utils.llm_client import llm_client
    
    if not llm_client.is_available:
        raise HTTPException(400, "LLM API가 설정되지 않았습니다.")
    
    translated, success = edit_translator.translate(trans_request.text)
    return {"success": success, "translated": translated}


@app.post("/api/edit/enhance")
async def enhance_edit_instruction(request: Request, enhance_request: EditEnhanceRequest):
    """편집 지시어 향상"""
    update_edit_activity()
    from utils.llm_client import llm_client
    
    if not llm_client.is_available:
        raise HTTPException(400, "LLM API가 설정되지 않았습니다.")
    
    enhanced, success = edit_enhancer.enhance(enhance_request.instruction)
    return {"success": success, "enhanced": enhanced}


@app.post("/api/edit/suggest")
async def suggest_edits(request: Request, suggest_request: EditSuggestRequest):
    """편집 아이디어 제안"""
    update_edit_activity()
    from utils.llm_client import llm_client
    
    if not llm_client.is_available:
        raise HTTPException(400, "LLM API가 설정되지 않았습니다.")
    
    suggestions, success = edit_suggester.suggest(
        context=suggest_request.context,
        image_description=suggest_request.image_description
    )
    
    # 한국어 제안도 함께 반환
    korean_suggestions, _ = edit_suggester.suggest_korean(
        context=suggest_request.context,
        image_description=suggest_request.image_description
    )
    
    return {
        "success": success,
        "suggestions": suggestions,
        "suggestions_korean": korean_suggestions
    }


# ============= 편집 히스토리 API =============
@app.get("/api/edit/history")
async def get_edit_history(request: Request):
    """편집 히스토리 목록 (사용자별)"""
    session = await get_session_from_request(request)
    require_auth(session)
    edit_history_mgr = get_edit_history_manager_sync(session.data_id)
    entries = edit_history_mgr.get_all()
    
    response = JSONResponse(content={"history": [e.to_dict() for e in entries[:50]]})
    set_session_cookie(response, session)
    return response


@app.get("/api/edit/history/{history_id}")
async def get_edit_history_detail(history_id: str, request: Request):
    """편집 히스토리 상세 정보 (사용자별)"""
    session = await get_session_from_request(request)
    require_auth(session)
    edit_history_mgr = get_edit_history_manager_sync(session.data_id)
    entry = edit_history_mgr.get_by_id(history_id)
    
    if not entry:
        raise HTTPException(404, "편집 히스토리를 찾을 수 없습니다.")
    
    return {"history": entry.to_dict()}


@app.get("/api/edit/history/{history_id}/chain")
async def get_edit_history_chain(history_id: str, request: Request):
    """멀티턴 편집 체인 가져오기 (사용자별)"""
    session = await get_session_from_request(request)
    require_auth(session)
    edit_history_mgr = get_edit_history_manager_sync(session.data_id)
    chain = edit_history_mgr.get_chain(history_id)
    
    return {"chain": [e.to_dict() for e in chain]}


@app.patch("/api/edit/history/{history_id}/conversation")
async def update_edit_history_conversation(history_id: str, request: Request, conv_request: EditConversationUpdateRequest):
    """편집 히스토리 대화 내용 업데이트 (사용자별)"""
    session = await get_session_from_request(request)
    require_auth(session)
    edit_history_mgr = get_edit_history_manager_sync(session.data_id)
    
    success = edit_history_mgr.update_conversation(history_id, conv_request.conversation)
    if not success:
        raise HTTPException(404, "편집 히스토리를 찾을 수 없습니다.")
    
    return {"success": True}


@app.delete("/api/edit/history")
async def clear_edit_history(request: Request):
    """편집 히스토리 삭제 (사용자별)"""
    session = await get_session_from_request(request)
    require_auth(session)
    edit_history_mgr = get_edit_history_manager_sync(session.data_id)
    edit_history_mgr.clear()
    return {"success": True}


@app.delete("/api/edit/history/{history_id}")
async def delete_edit_history_entry(history_id: str, request: Request):
    """편집 히스토리 항목 삭제 (사용자별)"""
    session = await get_session_from_request(request)
    require_auth(session)
    edit_history_mgr = get_edit_history_manager_sync(session.data_id)
    success = edit_history_mgr.delete(history_id)
    return {"success": success}


# ============= 메인 =============
if __name__ == "__main__":
    # 출력 폴더 생성
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("[*] Z-Image WebUI starting...")
    print(f"[*] http://localhost:{SERVER_PORT}")
    print("[*] Multi-user support enabled")
    
    uvicorn.run(
        app,
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=SERVER_RELOAD
    )
