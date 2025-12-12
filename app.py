"""Z-Image WebUI - FastAPI 기반 대화형 이미지 생성 웹앱 (다중 사용자 지원)"""

import os
import sys
import json
import asyncio
import base64
import random
import gc
import time
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
    SERVER_HOST,
    SERVER_PORT,
    SERVER_RELOAD,
    LONGCAT_EDIT_AUTO_UNLOAD_TIMEOUT,
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
from utils.longcat_edit import longcat_edit_manager
from utils.edit_history import get_edit_history_manager_sync, EditHistoryManager
from utils.edit_llm import edit_translator, edit_enhancer, edit_suggester


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
                # 모델 언로드
                del pipe
                pipe = None
                current_model = None
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
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
    model_path: str = ""
    cpu_offload: bool = False


class SettingsRequest(BaseModel):
    openai_api_key: str = ""  # 레거시 호환
    output_path: str = ""
    filename_pattern: str = "{date}_{time}_{seed}"
    # LLM Provider 설정
    llm_provider: str = ""
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""
    # 시스템 프롬프트 (번역/향상)
    translate_system_prompt: Optional[str] = None
    enhance_system_prompt: Optional[str] = None
    # 자동 언로드 설정
    auto_unload_enabled: Optional[bool] = None
    auto_unload_timeout: Optional[int] = None


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


# ============= 유틸리티 함수 =============
def get_device():
    """사용 가능한 디바이스 반환"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def image_to_base64(image: Image.Image) -> str:
    """PIL 이미지를 base64로 변환"""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def get_vram_info() -> str:
    """VRAM 사용량 정보"""
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / 1024**3
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"{vram_used:.1f}GB / {vram_total:.1f}GB"
    return "N/A"


async def get_session_from_request(request: Request) -> SessionInfo:
    """요청에서 세션 가져오기 또는 생성 - 클라이언트 IP 기반"""
    session_id = request.cookies.get(SessionManager.COOKIE_NAME)
    # 클라이언트 IP 가져오기
    client_host = request.client.host if request.client else None
    session = await session_manager.get_or_create_session(session_id, client_host)
    return session


def set_session_cookie(response: Response, session: SessionInfo):
    """응답에 세션 쿠키 설정"""
    response.set_cookie(
        key=SessionManager.COOKIE_NAME,
        value=session.session_id,
        max_age=SessionManager.COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax"
    )


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
    
    def get_session_id(self, websocket: WebSocket) -> Optional[str]:
        """WebSocket의 세션 ID 가져오기"""
        return self._websocket_sessions.get(websocket)


ws_manager = SessionConnectionManager()


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
    
    # 세션별 출력 디렉토리
    if session:
        outputs_dir = session.get_outputs_dir()
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
        
        # 동기 pipe 호출을 스레드에서 실행
        def run_pipe():
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
            "path": f"/outputs/{session_id}/{filename}" if session else f"/outputs/{filename}"
        })
    
    # 히스토리 추가 (세션별)
    if session:
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
    """메인 페이지"""
    update_activity()
    session = await get_session_from_request(request)
    
    response = templates.TemplateResponse("index.html", {"request": request})
    set_session_cookie(response, session)
    
    return response


@app.get("/api/status")
async def get_status(request: Request):
    """시스템 상태"""
    global pipe, current_model, device
    update_activity()
    
    session = await get_session_from_request(request)
    queue_status = generation_queue.get_queue_status()
    
    return {
        "model_loaded": pipe is not None,
        "current_model": current_model,
        "device": device or get_device(),
        "vram": get_vram_info(),
        "is_generating": queue_status["is_processing"],
        "upscaler_available": REALESRGAN_AVAILABLE,
        "queue_length": queue_status["queue_length"],
        "connected_users": ws_manager.get_session_count(),
        "session_id": session.session_id,
        "is_admin": is_localhost(request.client.host if request.client else None),
    }


@app.post("/api/model/load")
async def load_model(request: Request, model_request: ModelLoadRequest):
    """모델 로드"""
    global pipe, current_model, device, model_lock
    
    # 모델 잠금 확인
    if model_lock.locked():
        raise HTTPException(409, "다른 사용자가 모델을 로드/언로드 중입니다. 잠시 후 다시 시도해주세요.")
    
    async with model_lock:
        device = get_device()
        quant_info = QUANTIZATION_OPTIONS.get(model_request.quantization)
        
        if not quant_info:
            raise HTTPException(400, f"지원하지 않는 양자화: {model_request.quantization}")
        
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
                    cache_dir=model_request.model_path if model_request.model_path else None
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
                }
                if model_request.model_path:
                    load_kwargs["cache_dir"] = model_request.model_path
                
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
            
            if model_request.cpu_offload:
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
            
            current_model = model_request.quantization
            
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
                "current_model": current_model
            })
            
            await ws_manager.broadcast({
                "type": "complete",
                "content": f"✅ 모델 로드 완료! ({dtype}, {device})"
            })
            
            return {"success": True, "message": f"모델 로드 완료: {repo_id} ({dtype})"}
            
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
            
            del pipe
            pipe = None
            current_model = None
            
            await ws_manager.broadcast({
                "type": "model_progress", 
                "progress": 60, 
                "label": "VRAM 정리 중...",
                "detail": ""
            })
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
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
    
    if pipe is None:
        raise HTTPException(400, "모델이 로드되지 않았습니다.")
    
    if not gen_request.prompt.strip():
        raise HTTPException(400, "프롬프트를 입력해주세요.")
    
    # Rate limit 체크
    exceeded, count = session_manager.check_rate_limit(session.session_id)
    if exceeded:
        await ws_manager.send_to_session(session.session_id, {
            "type": "warning",
            "content": f"⚠️ 요청이 너무 많습니다. (분당 {count}회) 잠시 후 다시 시도해주세요."
        })
    
    # 요청 데이터 준비
    request_data = {
        "session_id": session.session_id,
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
    item_id, position = await generation_queue.add_to_queue(session.session_id, request_data)
    
    # 큐 상태 알림
    if position > 1:
        await ws_manager.send_to_session(session.session_id, {
            "type": "queue_status",
            "status": "queued",
            "position": position,
            "message": f"⏳ 대기열에 추가되었습니다. (순서: {position})"
        })
    else:
        await ws_manager.send_to_session(session.session_id, {
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
    """히스토리 목록 (세션별)"""
    session = await get_session_from_request(request)
    history_mgr = get_history_manager_sync(session.session_id)
    entries = history_mgr.get_all()
    
    response = JSONResponse(content={"history": [e.to_dict() for e in entries[:50]]})
    set_session_cookie(response, session)
    return response


@app.get("/api/history/{history_id}")
async def get_history_detail(history_id: str, request: Request):
    """히스토리 상세 정보 (세션별)"""
    session = await get_session_from_request(request)
    history_mgr = get_history_manager_sync(session.session_id)
    entry = history_mgr.get_by_id(history_id)
    
    if not entry:
        raise HTTPException(404, "히스토리를 찾을 수 없습니다.")
    
    return {"history": entry.to_dict()}


@app.patch("/api/history/{history_id}/conversation")
async def update_history_conversation(history_id: str, request: Request, conv_request: ConversationUpdateRequest):
    """히스토리의 대화 내용 업데이트 (세션별)"""
    session = await get_session_from_request(request)
    history_mgr = get_history_manager_sync(session.session_id)
    entry = history_mgr.get_by_id(history_id)
    
    if not entry:
        raise HTTPException(404, "히스토리를 찾을 수 없습니다.")
    
    entry.conversation = conv_request.conversation
    history_mgr._save()
    
    return {"success": True}


@app.delete("/api/history")
async def clear_history(request: Request):
    """히스토리 삭제 (세션별)"""
    session = await get_session_from_request(request)
    history_mgr = get_history_manager_sync(session.session_id)
    history_mgr.clear()
    return {"success": True}


# ============= 세션별 즐겨찾기 API =============
@app.get("/api/favorites")
async def get_favorites(request: Request):
    """즐겨찾기 목록 (세션별)"""
    session = await get_session_from_request(request)
    fav_mgr = get_favorites_manager_sync(session.session_id)
    entries = fav_mgr.get_all()
    
    response = JSONResponse(content={"favorites": [e.to_dict() for e in entries]})
    set_session_cookie(response, session)
    return response


@app.post("/api/favorites")
async def add_favorite(request: Request, fav_request: FavoriteRequest):
    """즐겨찾기 추가 (세션별)"""
    session = await get_session_from_request(request)
    fav_mgr = get_favorites_manager_sync(session.session_id)
    entry = fav_mgr.add(
        name=fav_request.name,
        prompt=fav_request.prompt,
        settings=fav_request.settings
    )
    return {"success": True, "id": entry.id}


@app.delete("/api/favorites/{fav_id}")
async def delete_favorite(fav_id: str, request: Request):
    """즐겨찾기 삭제 (세션별)"""
    session = await get_session_from_request(request)
    fav_mgr = get_favorites_manager_sync(session.session_id)
    success = fav_mgr.delete(fav_id)
    return {"success": success}


# ============= 세션별 갤러리 API =============
@app.get("/api/gallery")
async def get_gallery(request: Request):
    """갤러리 이미지 목록 (세션별)"""
    session = await get_session_from_request(request)
    outputs_dir = session.get_outputs_dir()
    
    images = []
    if outputs_dir.exists():
        for f in sorted(outputs_dir.glob("*.png"), key=lambda x: x.stat().st_mtime, reverse=True)[:50]:
            metadata = ImageMetadata.read_metadata(f)
            images.append({
                "filename": f.name,
                "path": f"/outputs/{session.session_id}/{f.name}",
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
        "quantization_options": list(QUANTIZATION_OPTIONS.keys()),
        "resolution_presets": RESOLUTION_PRESETS,
        # 자동 언로드 설정
        "auto_unload_enabled": settings.get("auto_unload_enabled", True),
        "auto_unload_timeout": settings.get("auto_unload_timeout", 10),
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


# ============= 관리자 API (localhost 전용) =============
@app.get("/api/admin/sessions")
async def get_all_sessions(request: Request):
    """모든 세션 목록 (관리자 전용)"""
    client_host = request.client.host if request.client else None
    if not is_localhost(client_host):
        raise HTTPException(403, "관리자 권한이 필요합니다.")
    
    sessions = session_manager.get_all_sessions()
    return {"sessions": sessions}


@app.delete("/api/admin/sessions/{session_id}")
async def delete_session(session_id: str, request: Request):
    """세션 삭제 (관리자 전용)"""
    client_host = request.client.host if request.client else None
    if not is_localhost(client_host):
        raise HTTPException(403, "관리자 권한이 필요합니다.")
    
    success = await session_manager.delete_session(session_id)
    return {"success": success}


# ============= WebSocket =============
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, z_image_session: Optional[str] = Cookie(default=None)):
    """웹소켓 연결 (세션별)"""
    # 세션 가져오기 - 클라이언트 IP 기반
    client_host = websocket.client.host if websocket.client else None
    session = await session_manager.get_or_create_session(z_image_session, client_host)
    
    await ws_manager.connect(websocket, session.session_id)
    update_activity()
    
    try:
        # 연결 시 상태 전송
        await websocket.send_json({
            "type": "connected",
            "content": "서버에 연결되었습니다.",
            "session_id": session.session_id,
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
    
    return {
        "model_loaded": longcat_edit_manager.is_loaded,
        "current_model": longcat_edit_manager.current_model,
        "device": longcat_edit_manager.device or longcat_edit_manager.get_device(),
        "vram": get_vram_info(),
        "session_id": session.session_id,
        "quantization_options": list(EDIT_QUANTIZATION_OPTIONS.keys()),
    }


@app.post("/api/edit/model/load")
async def load_edit_model(request: Request, model_request: EditModelLoadRequest):
    """LongCat-Image-Edit 모델 로드"""
    global edit_model_lock
    
    if edit_model_lock.locked():
        raise HTTPException(409, "다른 사용자가 모델을 로드/언로드 중입니다.")
    
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
            
            success, message = await longcat_edit_manager.load_model(
                quantization=model_request.quantization,
                cpu_offload=model_request.cpu_offload,
                model_path=model_request.model_path if model_request.model_path else None,
                progress_callback=progress_callback
            )
            
            if success:
                await ws_manager.broadcast({
                    "type": "edit_model_status_change",
                    "model_loaded": True,
                    "current_model": longcat_edit_manager.current_model
                })
                await ws_manager.broadcast({
                    "type": "complete",
                    "content": f"✅ 편집 모델 로드 완료!"
                })
                return {"success": True, "message": message}
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
                "type": "error",
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
            await ws_manager.send_to_session(session.session_id, {
                "type": "system",
                "content": "🌐 편집 지시어 번역 중..."
            })
            final_prompt, success = edit_translator.translate(prompt)
            if not success:
                await ws_manager.send_to_session(session.session_id, {
                    "type": "warning",
                    "content": "⚠️ 번역 실패, 원문 사용"
                })
        
        # 편집 시작 메시지
        await ws_manager.send_to_session(session.session_id, {
            "type": "system",
            "content": "🎨 이미지 편집 중..."
        })
        
        # 진행 상황 콜백 정의
        async def edit_progress_callback(current_image: int, total_images: int, steps: int):
            # 전체 진행률 계산 (이미지 기준)
            overall_progress = int((current_image - 1) / total_images * 100)
            
            await ws_manager.send_to_session(session.session_id, {
                "type": "edit_progress",
                "current_image": current_image,
                "total_images": total_images,
                "steps": steps,
                "progress": overall_progress
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
            progress_callback=edit_progress_callback
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
                "path": f"/outputs/{session.session_id}/{filename}"
            })
        
        # 히스토리 저장
        edit_history_mgr = get_edit_history_manager_sync(session.session_id)
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
        await ws_manager.send_to_session(session.session_id, {
            "type": "complete",
            "content": f"✅ 편집 완료! (시드: {results[0]['seed'] if results else 'N/A'})"
        })
        
        # 결과 전송
        await ws_manager.send_to_session(session.session_id, {
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
        await ws_manager.send_to_session(session.session_id, {
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
    """편집 히스토리 목록"""
    session = await get_session_from_request(request)
    edit_history_mgr = get_edit_history_manager_sync(session.session_id)
    entries = edit_history_mgr.get_all()
    
    response = JSONResponse(content={"history": [e.to_dict() for e in entries[:50]]})
    set_session_cookie(response, session)
    return response


@app.get("/api/edit/history/{history_id}")
async def get_edit_history_detail(history_id: str, request: Request):
    """편집 히스토리 상세 정보"""
    session = await get_session_from_request(request)
    edit_history_mgr = get_edit_history_manager_sync(session.session_id)
    entry = edit_history_mgr.get_by_id(history_id)
    
    if not entry:
        raise HTTPException(404, "편집 히스토리를 찾을 수 없습니다.")
    
    return {"history": entry.to_dict()}


@app.get("/api/edit/history/{history_id}/chain")
async def get_edit_history_chain(history_id: str, request: Request):
    """멀티턴 편집 체인 가져오기"""
    session = await get_session_from_request(request)
    edit_history_mgr = get_edit_history_manager_sync(session.session_id)
    chain = edit_history_mgr.get_chain(history_id)
    
    return {"chain": [e.to_dict() for e in chain]}


@app.patch("/api/edit/history/{history_id}/conversation")
async def update_edit_history_conversation(history_id: str, request: Request, conv_request: EditConversationUpdateRequest):
    """편집 히스토리 대화 내용 업데이트"""
    session = await get_session_from_request(request)
    edit_history_mgr = get_edit_history_manager_sync(session.session_id)
    
    success = edit_history_mgr.update_conversation(history_id, conv_request.conversation)
    if not success:
        raise HTTPException(404, "편집 히스토리를 찾을 수 없습니다.")
    
    return {"success": True}


@app.delete("/api/edit/history")
async def clear_edit_history(request: Request):
    """편집 히스토리 삭제"""
    session = await get_session_from_request(request)
    edit_history_mgr = get_edit_history_manager_sync(session.session_id)
    edit_history_mgr.clear()
    return {"success": True}


@app.delete("/api/edit/history/{history_id}")
async def delete_edit_history_entry(history_id: str, request: Request):
    """편집 히스토리 항목 삭제"""
    session = await get_session_from_request(request)
    edit_history_mgr = get_edit_history_manager_sync(session.session_id)
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
