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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form, Response, Cookie
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.requests import Request
from pydantic import BaseModel
import uvicorn

import torch
from PIL import Image

# 로컬 모듈
from config.defaults import (
    QUANTIZATION_OPTIONS,
    EDIT_QUANTIZATION_OPTIONS,
    RESOLUTION_PRESETS,
    OUTPUTS_DIR,
    MODELS_DIR,
    SERVER_HOST,
    SERVER_PORT,
    SERVER_RELOAD,
    LONGCAT_EDIT_AUTO_UNLOAD_TIMEOUT,
    DEFAULT_GPU_SETTINGS,
)
from config.templates import PROMPT_TEMPLATES
from utils.settings import settings
from utils.translator import translator
from utils.prompt_enhancer import prompt_enhancer
from utils.metadata import ImageMetadata, filename_generator
from utils.history import get_history_manager_sync, HistoryManager
from utils.favorites import get_favorites_manager_sync, FavoritesManager
from utils.upscaler import upscaler, REALESRGAN_AVAILABLE
from utils.session import session_manager, is_localhost, SessionManager, SessionInfo
from utils.queue_manager import generation_queue, GenerationQueueManager
from utils.auth import auth_manager, User
from utils.longcat_edit import longcat_edit_manager
from utils.edit_history import get_edit_history_manager_sync, EditHistoryManager
from utils.edit_llm import edit_translator, edit_enhancer, edit_suggester
from utils.gpu_monitor import gpu_monitor


# ============= 전역 변수 =============
pipe = None
current_model = None
device = None
last_activity_time = time.time()  # 마지막 활동 시간
auto_unload_task = None  # 자동 언로드 체크 태스크
model_lock = asyncio.Lock()  # 모델 로드/언로드 잠금

# LongCat-Image-Edit 관련
edit_last_activity_time = time.time()  # 편집 모델 마지막 활동 시간
edit_auto_unload_task = None  # 편집 모델 자동 언로드 태스크
edit_model_lock = asyncio.Lock()  # 편집 모델 로드/언로드 잠금


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
                
                print(f"✅ 자동 언로드 완료. VRAM: {get_vram_info()}")
                
            except Exception as e:
                print(f"❌ 자동 언로드 실패: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행되는 lifespan 핸들러"""
    global auto_unload_task
    
    # 시작 시: 자동 언로드 체크 태스크 시작
    auto_unload_task = asyncio.create_task(auto_unload_checker())
    print("🔄 자동 언로드 체커 시작됨")
    
    # 큐 워커 시작
    await generation_queue.start_worker()
    print("🔄 이미지 생성 큐 워커 시작됨")
    
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
app = FastAPI(title="Z-Image WebUI", version="2.0.0", lifespan=lifespan)

# 정적 파일 및 템플릿
app.mount("/static", StaticFiles(directory=ROOT_DIR / "static"), name="static")
templates = Jinja2Templates(directory=ROOT_DIR / "templates")


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
    # 편집 모델 설정 (관리자 전용)
    edit_quantization: Optional[str] = None
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
    quantization: str = "BF16 (기본, 최고품질)"
    model_path: str = ""
    cpu_offload: bool = True  # 기본 활성화 (VRAM 절약)
    target_device: str = "auto"  # 관리자 전용: "auto", "cuda:0", "cuda:1", "cpu", "mps"


class EditGenerateRequest(BaseModel):
    prompt: str
    korean_prompt: str = ""
    steps: int = 50
    guidance_scale: float = 4.5
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


# ============= 웹소켓 연결 관리 (세션별) =============
class SessionConnectionManager:
    """세션별 WebSocket 연결 관리"""
    
    def __init__(self):
        # session_id -> List[WebSocket]
        self._connections: Dict[str, List[WebSocket]] = {}
        self._websocket_sessions: Dict[WebSocket, str] = {}  # 역방향 매핑
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """연결 추가"""
        await websocket.accept()
        async with self._lock:
            if session_id not in self._connections:
                self._connections[session_id] = []
            self._connections[session_id].append(websocket)
            self._websocket_sessions[websocket] = session_id
    
    async def disconnect(self, websocket: WebSocket):
        """연결 제거"""
        async with self._lock:
            session_id = self._websocket_sessions.get(websocket)
            if session_id:
                if session_id in self._connections:
                    if websocket in self._connections[session_id]:
                        self._connections[session_id].remove(websocket)
                    # 세션의 모든 연결이 끊어지면 큐에서 제거
                    if not self._connections[session_id]:
                        del self._connections[session_id]
                        # 큐에서 해당 세션 요청 제거
                        await generation_queue.remove_session_items(session_id)
                del self._websocket_sessions[websocket]
    
    async def send_to_session(self, session_id: str, message: dict):
        """특정 세션에 메시지 전송"""
        async with self._lock:
            connections = self._connections.get(session_id, [])
            for ws in connections:
                try:
                    await ws.send_json(message)
                except:
                    pass
    
    async def broadcast(self, message: dict):
        """모든 연결에 브로드캐스트"""
        async with self._lock:
            for connections in self._connections.values():
                for ws in connections:
                    try:
                        await ws.send_json(message)
                    except:
                        pass
    
    def get_connection_count(self) -> int:
        """총 연결 수"""
        return sum(len(conns) for conns in self._connections.values())
    
    def get_session_count(self) -> int:
        """연결된 세션 수"""
        return len(self._connections)

    def get_connected_keys(self) -> List[str]:
        """현재 연결된 키 목록 (현재는 user_{id} 형태)"""
        return list(self._connections.keys())

    async def disconnect_key(self, key: str) -> int:
        """특정 키(user_{id})의 모든 WebSocket 연결 종료"""
        async with self._lock:
            connections = list(self._connections.get(key, []))
        closed = 0
        for ws in connections:
            try:
                await ws.close(code=4000)
                closed += 1
            except Exception:
                pass
        return closed
    
    def get_session_id(self, websocket: WebSocket) -> Optional[str]:
        """WebSocket의 세션 ID 가져오기"""
        return self._websocket_sessions.get(websocket)


ws_manager = SessionConnectionManager()


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
        }
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
    
    response = templates.TemplateResponse("login.html", {"request": request})
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
        "upscaler_available": REALESRGAN_AVAILABLE,
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


@app.post("/api/generate")
async def generate_image(request: Request, gen_request: GenerateRequest):
    """이미지 생성 요청 (큐에 추가)"""
    update_activity()
    
    session = await get_session_from_request(request)
    require_auth(session)
    
    if pipe is None:
        raise HTTPException(400, "모델이 로드되지 않았습니다.")
    
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
async def generate_preview(request: Request, gen_request: GenerateRequest):
    """빠른 미리보기 (256x256)"""
    gen_request.width = 256
    gen_request.height = 256
    gen_request.steps = min(gen_request.steps, 4)
    gen_request.num_images = 1
    return await generate_image(request, gen_request)


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


# ============= 세션별 히스토리 API =============
@app.get("/api/history")
async def get_history(request: Request):
    """히스토리 목록 (사용자별)"""
    session = await get_session_from_request(request)
    require_auth(session)
    history_mgr = get_history_manager_sync(session.data_id)
    entries = history_mgr.get_all()
    
    response = JSONResponse(content={"history": [e.to_dict() for e in entries[:50]]})
    set_session_cookie(response, session)
    return response


@app.get("/api/history/{history_id}")
async def get_history_detail(history_id: str, request: Request):
    """히스토리 상세 정보 (사용자별)"""
    session = await get_session_from_request(request)
    require_auth(session)
    history_mgr = get_history_manager_sync(session.data_id)
    entry = history_mgr.get_by_id(history_id)
    
    if not entry:
        raise HTTPException(404, "히스토리를 찾을 수 없습니다.")
    
    return {"history": entry.to_dict()}


@app.patch("/api/history/{history_id}/conversation")
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


@app.delete("/api/history")
async def clear_history(request: Request):
    """히스토리 삭제 (사용자별)"""
    session = await get_session_from_request(request)
    require_auth(session)
    history_mgr = get_history_manager_sync(session.data_id)
    history_mgr.clear()
    return {"success": True}


# ============= 사용자별 즐겨찾기 API =============
@app.get("/api/favorites")
async def get_favorites(request: Request):
    """즐겨찾기 목록 (사용자별)"""
    session = await get_session_from_request(request)
    require_auth(session)
    fav_mgr = get_favorites_manager_sync(session.data_id)
    entries = fav_mgr.get_all()
    
    response = JSONResponse(content={"favorites": [e.to_dict() for e in entries]})
    set_session_cookie(response, session)
    return response


@app.post("/api/favorites")
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


@app.delete("/api/favorites/{fav_id}")
async def delete_favorite(fav_id: str, request: Request):
    """즐겨찾기 삭제 (사용자별)"""
    session = await get_session_from_request(request)
    require_auth(session)
    fav_mgr = get_favorites_manager_sync(session.data_id)
    success = fav_mgr.delete(fav_id)
    return {"success": success}


# ============= 세션별 갤러리 API =============
@app.get("/api/gallery")
async def get_gallery(request: Request):
    """갤러리 이미지 목록 (사용자별)"""
    session = await get_session_from_request(request)
    require_auth(session)
    outputs_dir = session.get_outputs_dir()
    
    images = []
    if outputs_dir.exists():
        for f in sorted(outputs_dir.glob("*.png"), key=lambda x: x.stat().st_mtime, reverse=True)[:50]:
            metadata = ImageMetadata.read_metadata(f)
            images.append({
                "filename": f.name,
                "path": f"/outputs/{session.data_id}/{f.name}",
                "metadata": metadata
            })
    
    response = JSONResponse(content={"images": images})
    set_session_cookie(response, session)
    return response


# ============= 설정 API (localhost 전용 쓰기) =============
@app.post("/api/settings")
async def save_settings(request: Request, settings_request: SettingsRequest):
    """설정 저장 (localhost만 허용)"""
    # localhost 체크
    client_host = request.client.host if request.client else None
    if not is_localhost(client_host):
        raise HTTPException(403, "설정 변경은 localhost에서만 가능합니다.")
    
    from utils.llm_client import llm_client
    
    # 레거시 호환
    if settings_request.openai_api_key:
        settings.set("openai_api_key", settings_request.openai_api_key)
        translator.set_api_key(settings_request.openai_api_key)
        prompt_enhancer.set_api_key(settings_request.openai_api_key)
    
    # LLM Provider 설정
    if settings_request.llm_provider:
        settings.set("llm_provider", settings_request.llm_provider)
    
    if settings_request.llm_api_key:
        settings.set("llm_api_key", settings_request.llm_api_key)
        settings.set("openai_api_key", settings_request.llm_api_key)
    
    if settings_request.llm_base_url is not None:
        settings.set("llm_base_url", settings_request.llm_base_url)
    
    if settings_request.llm_model is not None:
        settings.set("llm_model", settings_request.llm_model)
    
    llm_client.invalidate()
    
    if settings_request.output_path:
        settings.set("output_path", settings_request.output_path)
    
    if settings_request.filename_pattern:
        settings.set("filename_pattern", settings_request.filename_pattern)
    
    if settings_request.translate_system_prompt is not None:
        settings.set("translate_system_prompt", settings_request.translate_system_prompt)
    
    if settings_request.enhance_system_prompt is not None:
        settings.set("enhance_system_prompt", settings_request.enhance_system_prompt)
    
    if settings_request.auto_unload_enabled is not None:
        settings.set("auto_unload_enabled", settings_request.auto_unload_enabled)
    
    if settings_request.auto_unload_timeout is not None:
        timeout = max(1, min(1440, settings_request.auto_unload_timeout))
        settings.set("auto_unload_timeout", timeout)
    
    if settings_request.edit_auto_unload_enabled is not None:
        settings.set("edit_auto_unload_enabled", settings_request.edit_auto_unload_enabled)
    
    if settings_request.edit_auto_unload_timeout is not None:
        timeout = max(1, min(1440, settings_request.edit_auto_unload_timeout))
        settings.set("edit_auto_unload_timeout", timeout)

    # 모델 설정 (관리자 전용)
    if settings_request.quantization is not None:
        if settings_request.quantization not in QUANTIZATION_OPTIONS:
            raise HTTPException(400, f"지원하지 않는 양자화: {settings_request.quantization}")
        settings.set("quantization", settings_request.quantization)

    if settings_request.cpu_offload is not None:
        settings.set("cpu_offload", bool(settings_request.cpu_offload))

    # 편집 모델 설정 (관리자 전용)
    if settings_request.edit_quantization is not None:
        if settings_request.edit_quantization not in EDIT_QUANTIZATION_OPTIONS:
            raise HTTPException(400, f"지원하지 않는 편집 양자화: {settings_request.edit_quantization}")
        settings.set("edit_quantization", settings_request.edit_quantization)

    if settings_request.edit_cpu_offload is not None:
        settings.set("edit_cpu_offload", bool(settings_request.edit_cpu_offload))
    
    return {"success": True}


@app.get("/api/settings")
async def get_settings(request: Request):
    """설정 가져오기"""
    from utils.settings import LLM_PROVIDERS
    from utils.translator import Translator
    from utils.prompt_enhancer import PromptEnhancer
    from utils.edit_llm import EditTranslator, EditEnhancer, EditSuggester
    
    session = await get_session_from_request(request)
    client_host = request.client.host if request.client else None
    is_admin = is_localhost(client_host)
    
    # 세션별 시스템 프롬프트 (개인화)
    session_translate_prompt = session.get_setting("translate_system_prompt")
    session_enhance_prompt = session.get_setting("enhance_system_prompt")
    
    # 세션에 설정이 없으면 전역 설정 사용, 전역도 없으면 기본값
    translate_prompt = session_translate_prompt or settings.get("translate_system_prompt") or Translator.DEFAULT_SYSTEM_PROMPT
    enhance_prompt = session_enhance_prompt or settings.get("enhance_system_prompt") or PromptEnhancer.DEFAULT_SYSTEM_PROMPT
    
    # 편집 시스템 프롬프트 (세션별 개인화)
    session_edit_translate = session.get_setting("edit_translate_system_prompt")
    session_edit_enhance = session.get_setting("edit_enhance_system_prompt")
    session_edit_suggest = session.get_setting("edit_suggest_system_prompt")
    
    edit_translate_prompt = session_edit_translate or settings.get("edit_translate_system_prompt") or EditTranslator.DEFAULT_SYSTEM_PROMPT
    edit_enhance_prompt = session_edit_enhance or settings.get("edit_enhance_system_prompt") or EditEnhancer.DEFAULT_SYSTEM_PROMPT
    edit_suggest_prompt = session_edit_suggest or settings.get("edit_suggest_system_prompt") or EditSuggester.DEFAULT_SYSTEM_PROMPT
    
    return {
        # 관리자 여부
        "is_admin": is_admin,
        # 레거시 호환
        "openai_api_key": "***" if settings.get("openai_api_key") else "",
        # LLM Provider 설정
        "llm_provider": settings.get("llm_provider", "env"),
        "llm_api_key": "***" if settings.get("llm_api_key") else "",
        "llm_base_url": settings.get("llm_base_url", ""),
        "llm_model": settings.get("llm_model", ""),
        "llm_providers": {
            pid: {
                "name": pinfo["name"],
                "base_url": pinfo["base_url"],
                "default_model": pinfo["default_model"],
                "models": pinfo["models"],
            }
            for pid, pinfo in LLM_PROVIDERS.items()
        },
        # 시스템 프롬프트 (세션별 개인화)
        "translate_system_prompt": translate_prompt,
        "enhance_system_prompt": enhance_prompt,
        "default_translate_system_prompt": Translator.DEFAULT_SYSTEM_PROMPT,
        "default_enhance_system_prompt": PromptEnhancer.DEFAULT_SYSTEM_PROMPT,
        # 편집 시스템 프롬프트 (세션별 개인화)
        "edit_translate_system_prompt": edit_translate_prompt,
        "edit_enhance_system_prompt": edit_enhance_prompt,
        "edit_suggest_system_prompt": edit_suggest_prompt,
        "default_edit_translate_system_prompt": EditTranslator.DEFAULT_SYSTEM_PROMPT,
        "default_edit_enhance_system_prompt": EditEnhancer.DEFAULT_SYSTEM_PROMPT,
        "default_edit_suggest_system_prompt": EditSuggester.DEFAULT_SYSTEM_PROMPT,
        # 기타 설정
        "output_path": str(settings.get("output_path", OUTPUTS_DIR)),
        "filename_pattern": settings.get("filename_pattern", "{date}_{time}_{seed}"),
        # 모델 설정 (관리자만 변경 가능 - 모든 사용자에게는 현재 값만 제공)
        "quantization": settings.get("quantization", "BF16 (기본, 최고품질)"),
        "cpu_offload": settings.get("cpu_offload", False),
        "edit_quantization": settings.get("edit_quantization", "BF16 (기본, 최고품질)"),
        "edit_cpu_offload": settings.get("edit_cpu_offload", True),
        "quantization_options": list(QUANTIZATION_OPTIONS.keys()),
        "resolution_presets": RESOLUTION_PRESETS,
        # 자동 언로드 설정
        "auto_unload_enabled": settings.get("auto_unload_enabled", True),
        "auto_unload_timeout": settings.get("auto_unload_timeout", 10),
        # 편집 모델 자동 언로드 설정
        "edit_auto_unload_enabled": settings.get("edit_auto_unload_enabled", True),
        "edit_auto_unload_timeout": settings.get("edit_auto_unload_timeout", LONGCAT_EDIT_AUTO_UNLOAD_TIMEOUT),
    }


# ============= 세션별 시스템 프롬프트 API =============
class SystemPromptsRequest(BaseModel):
    translate_system_prompt: Optional[str] = None
    enhance_system_prompt: Optional[str] = None
    # 편집 시스템 프롬프트 (개인화)
    edit_translate_system_prompt: Optional[str] = None
    edit_enhance_system_prompt: Optional[str] = None
    edit_suggest_system_prompt: Optional[str] = None


@app.post("/api/settings/prompts")
async def save_session_prompts(request: Request, prompts_request: SystemPromptsRequest):
    """시스템 프롬프트 저장 (세션별 개인화, 모든 사용자 접근 가능)"""
    session = await get_session_from_request(request)
    session_settings = session.get_settings()
    
    # 생성 시스템 프롬프트
    if prompts_request.translate_system_prompt is not None:
        if prompts_request.translate_system_prompt == '':
            # 빈 문자열이면 설정 삭제 (기본값 사용)
            session_settings.pop("translate_system_prompt", None)
        else:
            session_settings["translate_system_prompt"] = prompts_request.translate_system_prompt
    
    if prompts_request.enhance_system_prompt is not None:
        if prompts_request.enhance_system_prompt == '':
            # 빈 문자열이면 설정 삭제 (기본값 사용)
            session_settings.pop("enhance_system_prompt", None)
        else:
            session_settings["enhance_system_prompt"] = prompts_request.enhance_system_prompt
    
    # 편집 시스템 프롬프트
    if prompts_request.edit_translate_system_prompt is not None:
        if prompts_request.edit_translate_system_prompt == '':
            session_settings.pop("edit_translate_system_prompt", None)
        else:
            session_settings["edit_translate_system_prompt"] = prompts_request.edit_translate_system_prompt
    
    if prompts_request.edit_enhance_system_prompt is not None:
        if prompts_request.edit_enhance_system_prompt == '':
            session_settings.pop("edit_enhance_system_prompt", None)
        else:
            session_settings["edit_enhance_system_prompt"] = prompts_request.edit_enhance_system_prompt
    
    if prompts_request.edit_suggest_system_prompt is not None:
        if prompts_request.edit_suggest_system_prompt == '':
            session_settings.pop("edit_suggest_system_prompt", None)
        else:
            session_settings["edit_suggest_system_prompt"] = prompts_request.edit_suggest_system_prompt
    
    session.save_settings(session_settings)
    
    response = JSONResponse(content={"success": True})
    set_session_cookie(response, session)
    return response


@app.delete("/api/settings/prompts")
async def reset_session_prompts(request: Request):
    """시스템 프롬프트 초기화 (세션별 설정 삭제, 전역/기본값 사용)"""
    session = await get_session_from_request(request)
    
    session_settings = session.get_settings()
    # 생성 시스템 프롬프트
    if "translate_system_prompt" in session_settings:
        del session_settings["translate_system_prompt"]
    if "enhance_system_prompt" in session_settings:
        del session_settings["enhance_system_prompt"]
    # 편집 시스템 프롬프트
    if "edit_translate_system_prompt" in session_settings:
        del session_settings["edit_translate_system_prompt"]
    if "edit_enhance_system_prompt" in session_settings:
        del session_settings["edit_enhance_system_prompt"]
    if "edit_suggest_system_prompt" in session_settings:
        del session_settings["edit_suggest_system_prompt"]
    session.save_settings(session_settings)
    
    response = JSONResponse(content={"success": True})
    set_session_cookie(response, session)
    return response


# ============= 인증 API =============
@app.post("/api/auth/register")
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


@app.post("/api/auth/login")
async def login(request: Request, data: LoginRequest):
    """로그인"""
    # 로그인은 세션(로그인 쿠키) 발급이 필요하므로 생성 허용
    session = await get_session_from_request(request, create_if_missing=True)
    
    # 인증
    success, message, user = auth_manager.authenticate(data.username, data.password)
    
    if not success or not user:
        raise HTTPException(401, message)
    
    # 세션에 로그인 정보 연결
    await session_manager.login_session(session.session_id, user.id, user.username)
    
    response = JSONResponse(content={
        "success": True,
        "message": message,
        "user": user.to_dict()
    })
    set_session_cookie(response, session)
    return response


@app.post("/api/auth/logout")
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


@app.get("/api/auth/me")
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


@app.post("/api/auth/change-password")
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


# ============= 관리자 API (localhost 전용) =============
@app.get("/api/admin/users")
async def get_all_users(request: Request):
    """모든 사용자 목록 (관리자 전용)"""
    require_admin(request)
    
    users = auth_manager.get_all_users()
    return {"users": users}


@app.post("/api/admin/users/{user_id}/reset-password")
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


@app.delete("/api/admin/users/{user_id}")
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


@app.get("/api/admin/sessions")
async def get_all_sessions(request: Request):
    """접속 사용자(계정) 목록 (관리자 전용)"""
    require_admin(request)

    users = []
    # WebSocket 연결 키는 현재 계정(data_id)로 통일되어 있음
    for data_id in sorted(ws_manager.get_connected_keys()):
        user_id = _parse_user_id_from_data_id(data_id)
        user = auth_manager.get_user_by_id(user_id) if user_id is not None else None
        username = user.username if user else None

        # last_activity는 (있다면) 해당 유저의 활성 세션에서 가져옴
        session = session_manager.get_session_by_user(user_id) if user_id is not None else None
        last_activity = session.last_activity if session else None

        users.append({
            "data_id": data_id,           # user_{id}
            "user_id": user_id,
            "username": username,
            "last_activity": last_activity,
            "data_size": _get_data_size_by_data_id(data_id),
            "connected": True,
        })

    return {"users": users}


@app.delete("/api/admin/sessions/{session_id}")
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


# ============= GPU 관리 API (관리자 전용) =============
class GPUSettingsRequest(BaseModel):
    generation_gpu: Optional[str] = None  # "auto", "cuda:0", "cuda:1", "cpu"
    edit_gpu: Optional[str] = None        # "auto", "cuda:0", "cuda:1", "cpu"


@app.get("/api/admin/gpu-status")
async def get_gpu_status(request: Request):
    """GPU 상태 조회 (관리자 전용)"""
    require_admin(request)
    
    gpu_info = gpu_monitor.get_system_info()
    
    # 현재 모델 상태 추가
    gpu_info["models"] = {
        "generation": {
            "loaded": pipe is not None,
            "name": current_model,
            "device": device,
        },
        "edit": {
            "loaded": longcat_edit_manager.is_loaded,
            "name": longcat_edit_manager.current_model,
            "device": longcat_edit_manager.device,
            "quantization": longcat_edit_manager.current_quantization,
            "cpu_offload": longcat_edit_manager.cpu_offload_enabled,
        }
    }
    
    # 현재 GPU 설정 추가
    gpu_info["current_settings"] = {
        "generation_gpu": settings.get("generation_gpu", DEFAULT_GPU_SETTINGS["generation_gpu"]),
        "edit_gpu": settings.get("edit_gpu", DEFAULT_GPU_SETTINGS["edit_gpu"]),
    }
    
    return gpu_info


@app.post("/api/admin/gpu-settings")
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


@app.get("/api/admin/available-devices")
async def get_available_devices(request: Request):
    """사용 가능한 디바이스 목록 (관리자 전용)"""
    require_admin(request)
    
    return {
        "devices": gpu_monitor.get_available_devices(),
        "gpu_count": gpu_monitor.gpu_count,
        "cuda_available": gpu_monitor.cuda_available,
    }


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
            "model_loaded": longcat_edit_manager.is_loaded,
            "current_model": longcat_edit_manager.current_model
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


# ============= LongCat-Image-Edit API =============

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
        "model_loaded": longcat_edit_manager.is_loaded,
        "current_model": longcat_edit_manager.current_model,
        "current_quantization": longcat_edit_manager.current_quantization,
        "cpu_offload_enabled": longcat_edit_manager.cpu_offload_enabled,
        # 저장된(기본) 편집 모델 설정값 - 새로고침/재시작 후 UI에서 유지되도록 제공
        "saved_edit_quantization": settings.get("edit_quantization", "BF16 (기본, 최고품질)"),
        "saved_edit_cpu_offload": settings.get("edit_cpu_offload", True),
        "device": longcat_edit_manager.device or longcat_edit_manager.get_device(),
        "vram": get_vram_info(),
        "session_id": session.data_id,
        "is_admin": is_admin,
        "quantization_options": list(EDIT_QUANTIZATION_OPTIONS.keys()),
        # 양자화 옵션 상세 정보 (예상 VRAM 포함)
        "quantization_details": {
            name: {
                "type": info.get("type"),
                "estimated_vram": info.get("estimated_vram", "N/A"),
            }
            for name, info in EDIT_QUANTIZATION_OPTIONS.items()
        },
    }
    
    # 관리자인 경우 추가 정보
    if is_admin:
        status["available_devices"] = gpu_monitor.get_available_devices()
    
    return status


@app.post("/api/edit/model/load")
async def load_edit_model(request: Request, model_request: EditModelLoadRequest):
    """LongCat-Image-Edit 모델 로드"""
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
            
            # 양자화/CPU 오프로딩은 관리자만 변경 가능
            requested_quantization = model_request.quantization
            requested_cpu_offload = model_request.cpu_offload
            if not is_admin:
                requested_quantization = settings.get("edit_quantization", requested_quantization)
                requested_cpu_offload = settings.get("edit_cpu_offload", requested_cpu_offload)

            success, message = await longcat_edit_manager.load_model(
                quantization=requested_quantization,
                cpu_offload=requested_cpu_offload,
                model_path=model_request.model_path if model_request.model_path else None,
                target_device=target_device,
                progress_callback=progress_callback
            )
            
            if success:
                await ws_manager.broadcast({
                    "type": "edit_model_status_change",
                    "model_loaded": True,
                    "current_model": longcat_edit_manager.current_model,
                    "device": longcat_edit_manager.device
                })
                await ws_manager.broadcast({
                    "type": "edit_system",
                    "content": f"✅ 편집 모델 로드 완료! ({longcat_edit_manager.device})"
                })
                return {"success": True, "message": message, "device": longcat_edit_manager.device}
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
    """LongCat-Image-Edit 모델 언로드"""
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
            
            success, message = await longcat_edit_manager.unload_model()
            
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


@app.post("/api/edit/generate")
async def edit_image(
    request: Request,
    image: UploadFile = File(...),
    prompt: str = Form(...),
    korean_prompt: str = Form(""),
    steps: int = Form(50),
    guidance_scale: float = Form(4.5),
    seed: int = Form(-1),
    num_images: int = Form(1),
    auto_translate: str = Form("true"),
    reference_image: Optional[UploadFile] = File(None)
):
    """이미지 편집 실행"""
    update_edit_activity()
    
    session = await get_session_from_request(request)
    require_auth(session)
    
    if not longcat_edit_manager.is_loaded:
        raise HTTPException(400, "편집 모델이 로드되지 않았습니다.")
    
    if not prompt.strip():
        raise HTTPException(400, "편집 프롬프트를 입력해주세요.")
    
    # Form에서 받은 auto_translate 문자열을 bool로 변환
    auto_translate_bool = auto_translate.lower() in ("true", "1", "yes")
    
    try:
        # 이미지 로드
        image_data = await image.read()
        pil_image = Image.open(BytesIO(image_data)).convert("RGB")
        
        # 참조 이미지 로드 (있으면)
        ref_image = None
        if reference_image:
            ref_data = await reference_image.read()
            ref_image = Image.open(BytesIO(ref_data)).convert("RGB")
        
        # 프롬프트 번역
        final_prompt = prompt
        if auto_translate_bool and edit_translator.is_korean(prompt):
            await ws_manager.send_to_session(session.data_id, {
                "type": "edit_system",
                "content": "🌐 편집 지시어 번역 중..."
            })
            final_prompt, success = edit_translator.translate(prompt)
            if not success:
                await ws_manager.send_to_session(session.data_id, {
                    "type": "edit_system",
                    "content": "⚠️ 번역 실패, 원문 사용"
                })
        
        # 편집 시작 메시지
        await ws_manager.send_to_session(session.data_id, {
            "type": "edit_system",
            "content": "🎨 이미지 편집 중..."
        })
        
        # 진행 상황 콜백 정의
        async def edit_progress_callback(current_image: int, total_images: int, current_step: int, total_steps: int):
            # 전체 진행률 계산 (이미지 + 스텝 기준)
            image_progress = (current_image - 1) / total_images
            step_progress = current_step / total_steps / total_images
            overall_progress = int((image_progress + step_progress) * 100)
            
            await ws_manager.send_to_session(session.data_id, {
                "type": "edit_progress",
                "current_image": current_image,
                "total_images": total_images,
                "current_step": current_step,
                "total_steps": total_steps,
                "progress": overall_progress
            })
        
        # 상태 메시지 콜백 정의 (참조 이미지 분석 등)
        async def edit_status_callback(message: str):
            await ws_manager.send_to_session(session.data_id, {
                "type": "edit_system",
                "content": message
            })
        
        # 편집 실행
        success, results, message = await longcat_edit_manager.edit_image(
            image=pil_image,
            prompt=final_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            num_images=num_images,
            reference_image=ref_image,
            progress_callback=edit_progress_callback,
            status_callback=edit_status_callback
        )
        
        if not success:
            raise HTTPException(500, message)
        
        # 세션별 출력 디렉토리
        outputs_dir = session.get_outputs_dir()
        outputs_dir.mkdir(parents=True, exist_ok=True)
        
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
                model="LongCat-Image-Edit",
            )
            ImageMetadata.save_with_metadata(result_image, output_path, metadata)
            
            result_paths.append(str(output_path))
            images_response.append({
                "base64": image_to_base64(result_image),
                "filename": filename,
                "seed": seed,
                "path": f"/outputs/{session.data_id}/{filename}"
            })
        
        # 히스토리 저장
        edit_history_mgr = get_edit_history_manager_sync(session.data_id)
        history_entry = edit_history_mgr.add(
            prompt=final_prompt,
            korean_prompt=korean_prompt,
            settings={
                "steps": steps,
                "guidance_scale": guidance_scale,
                "seed": results[0]["seed"] if results else -1,
            },
            result_image_paths=[img["path"] for img in images_response]
        )
        
        # 완료 메시지
        await ws_manager.send_to_session(session.data_id, {
            "type": "edit_system",
            "content": f"✅ 편집 완료! (시드: {results[0]['seed'] if results else 'N/A'})"
        })
        
        # 결과 전송
        await ws_manager.send_to_session(session.data_id, {
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
        await ws_manager.send_to_session(session.data_id, {
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
    
    print("🎨 Z-Image WebUI 시작...")
    print(f"📍 http://localhost:{SERVER_PORT}")
    print("🌐 다중 사용자 지원 활성화")
    
    uvicorn.run(
        app,
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=SERVER_RELOAD
    )
