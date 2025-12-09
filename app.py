"""Z-Image WebUI - FastAPI 기반 대화형 이미지 생성 웹앱"""

import os
import sys
import json
import asyncio
import base64
import random
import gc
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from io import BytesIO

# 프로젝트 루트를 path에 추가
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
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
    RESOLUTION_PRESETS,
    OUTPUTS_DIR,
)
from config.templates import PROMPT_TEMPLATES
from utils.settings import settings
from utils.translator import translator
from utils.prompt_enhancer import prompt_enhancer
from utils.metadata import ImageMetadata, filename_generator
from utils.history import history_manager
from utils.favorites import favorites_manager
from utils.upscaler import upscaler, REALESRGAN_AVAILABLE


# ============= FastAPI 앱 설정 =============
app = FastAPI(title="Z-Image WebUI", version="1.0.0")

# 정적 파일 및 템플릿
app.mount("/static", StaticFiles(directory=ROOT_DIR / "static"), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
templates = Jinja2Templates(directory=ROOT_DIR / "templates")


# ============= 전역 변수 =============
pipe = None
current_model = None
device = None
is_generating = False


# ============= Pydantic 모델 =============
class GenerateRequest(BaseModel):
    prompt: str
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


class FavoriteRequest(BaseModel):
    name: str
    prompt: str
    settings: dict = {}


class TranslateRequest(BaseModel):
    text: str


class EnhanceRequest(BaseModel):
    prompt: str
    style: str = "기본"


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


# ============= 웹소켓 연결 관리 =============
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


# ============= API 엔드포인트 =============

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """메인 페이지"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/status")
async def get_status():
    """시스템 상태"""
    global pipe, current_model, device
    return {
        "model_loaded": pipe is not None,
        "current_model": current_model,
        "device": device or get_device(),
        "vram": get_vram_info(),
        "is_generating": is_generating,
        "upscaler_available": REALESRGAN_AVAILABLE,
    }


@app.post("/api/model/load")
async def load_model(request: ModelLoadRequest):
    """모델 로드"""
    global pipe, current_model, device
    
    device = get_device()
    quant_info = QUANTIZATION_OPTIONS.get(request.quantization)
    
    if not quant_info:
        raise HTTPException(400, f"지원하지 않는 양자화: {request.quantization}")
    
    repo_id = quant_info["repo"]
    dtype = quant_info["type"]
    is_gguf = quant_info.get("is_gguf", False)
    
    try:
        # 1단계: 로딩 준비
        await manager.broadcast({
            "type": "model_progress", 
            "progress": 5, 
            "label": "🔧 모델 초기화 중...",
            "detail": f"양자화: {dtype}, 디바이스: {device}",
            "stage": "init"
        })
        await asyncio.sleep(0.1)  # 메시지 전송 대기
        
        from diffusers import ZImagePipeline
        
        if is_gguf:
            # GGUF 양자화 모델 로드
            from diffusers import ZImageTransformer2DModel, GGUFQuantizationConfig
            from huggingface_hub import hf_hub_download
            
            filename = quant_info["filename"]
            
            # 2단계: GGUF 다운로드
            await manager.broadcast({
                "type": "model_progress", 
                "progress": 10, 
                "label": "📥 GGUF 모델 다운로드 확인 중...",
                "detail": f"파일: {filename} (캐시 확인 중...)",
                "stage": "download"
            })
            await asyncio.sleep(0.1)
            
            # GGUF 파일 다운로드 (캐시됨)
            gguf_path = await asyncio.to_thread(
                hf_hub_download,
                repo_id=repo_id, 
                filename=filename,
                cache_dir=request.model_path if request.model_path else None
            )
            
            # 3단계: GGUF Transformer 로드
            await manager.broadcast({
                "type": "model_progress", 
                "progress": 30, 
                "label": "🔄 GGUF Transformer 로딩 중...",
                "detail": f"양자화 타입: {dtype} (시간이 걸릴 수 있습니다)",
                "stage": "load_transformer"
            })
            await asyncio.sleep(0.1)
            
            # GGUF Transformer 로드 (동기 작업을 스레드에서 실행)
            transformer = await asyncio.to_thread(
                ZImageTransformer2DModel.from_single_file,
                gguf_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                torch_dtype=torch.bfloat16,
            )
            
            # 4단계: 파이프라인 구성
            await manager.broadcast({
                "type": "model_progress", 
                "progress": 55, 
                "label": "🔗 파이프라인 구성 중...",
                "detail": "기본 모델 다운로드/로드 및 GGUF Transformer 결합",
                "stage": "load_pipeline"
            })
            await asyncio.sleep(0.1)
            
            # 파이프라인 구성 (GGUF transformer 사용)
            pipe = await asyncio.to_thread(
                ZImagePipeline.from_pretrained,
                "Tongyi-MAI/Z-Image-Turbo",
                transformer=transformer,
                torch_dtype=torch.bfloat16,
            )
        else:
            # 기본 BF16 모델 로드
            # 2단계: 다운로드/로드
            await manager.broadcast({
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
            if request.model_path:
                load_kwargs["cache_dir"] = request.model_path
            
            # 3단계: 모델 파일 로딩
            await manager.broadcast({
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
        await manager.broadcast({
            "type": "model_progress", 
            "progress": 75, 
            "label": f"🚀 {device.upper()}로 모델 전송 중...",
            "detail": "VRAM으로 모델 복사 중... (VRAM 크기에 따라 시간이 걸립니다)",
            "stage": "to_device"
        })
        await asyncio.sleep(0.1)
        
        if request.cpu_offload:
            await asyncio.to_thread(pipe.enable_model_cpu_offload)
            await manager.broadcast({
                "type": "model_progress", 
                "progress": 95, 
                "label": "⚙️ CPU 오프로딩 설정 중...",
                "detail": "VRAM 부족 시 자동으로 RAM 사용",
                "stage": "cpu_offload"
            })
        else:
            await asyncio.to_thread(pipe.to, device)
        
        current_model = request.quantization
        
        # 6단계: 완료
        await manager.broadcast({
            "type": "model_progress", 
            "progress": 100, 
            "label": "✅ 모델 로드 완료!",
            "detail": f"VRAM 사용량: {get_vram_info()}",
            "stage": "complete"
        })
        
        await manager.broadcast({
            "type": "complete",
            "content": f"✅ 모델 로드 완료! ({dtype}, {device})"
        })
        
        return {"success": True, "message": f"모델 로드 완료: {repo_id} ({dtype})"}
        
    except Exception as e:
        await manager.broadcast({
            "type": "model_progress", 
            "progress": 0, 
            "label": "❌ 로드 실패",
            "detail": str(e),
            "stage": "error"
        })
        await manager.broadcast({"type": "error", "content": f"❌ 모델 로드 실패: {str(e)}"})
        raise HTTPException(500, str(e))


@app.post("/api/model/unload")
async def unload_model():
    """모델 언로드"""
    global pipe, current_model
    
    if pipe is None:
        return {"success": True, "message": "로드된 모델이 없습니다."}
    
    try:
        await manager.broadcast({
            "type": "model_progress", 
            "progress": 30, 
            "label": "모델 메모리 해제 중...",
            "detail": ""
        })
        
        del pipe
        pipe = None
        current_model = None
        
        await manager.broadcast({
            "type": "model_progress", 
            "progress": 60, 
            "label": "VRAM 정리 중...",
            "detail": ""
        })
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        gc.collect()
        
        await manager.broadcast({
            "type": "model_progress", 
            "progress": 100, 
            "label": "언로드 완료!",
            "detail": f"VRAM 사용량: {get_vram_info()}"
        })
        
        await manager.broadcast({"type": "complete", "content": "✅ 모델 언로드 완료!"})
        return {"success": True, "message": "모델 언로드 완료"}
        
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/generate")
async def generate_image(request: GenerateRequest):
    """이미지 생성"""
    global pipe, is_generating
    
    if pipe is None:
        raise HTTPException(400, "모델이 로드되지 않았습니다.")
    
    if is_generating:
        raise HTTPException(400, "이미 생성 중입니다.")
    
    if not request.prompt.strip():
        raise HTTPException(400, "프롬프트를 입력해주세요.")
    
    is_generating = True
    
    try:
        # 번역
        final_prompt = request.prompt
        if request.auto_translate and translator.is_korean(request.prompt):
            await manager.broadcast({"type": "system", "content": "🌐 프롬프트 번역 중..."})
            final_prompt, success = translator.translate(request.prompt)
            if not success:
                await manager.broadcast({"type": "warning", "content": "⚠️ 번역 실패, 원문 사용"})
        
        # 시드 설정
        seed = request.seed if request.seed != -1 else random.randint(0, 2147483647)
        
        images = []
        for i in range(request.num_images):
            current_seed = seed + i
            await manager.broadcast({
                "type": "progress",
                "content": f"🎨 이미지 생성 중... ({i+1}/{request.num_images})"
            })
            
            generator = torch.Generator(device).manual_seed(current_seed)
            
            image = pipe(
                prompt=final_prompt,
                height=request.height,
                width=request.width,
                num_inference_steps=request.steps,
                guidance_scale=request.guidance_scale,
                generator=generator,
            ).images[0]
            
            # 메타데이터 생성 및 저장
            metadata = ImageMetadata.create_metadata(
                prompt=final_prompt,
                seed=current_seed,
                width=request.width,
                height=request.height,
                steps=request.steps,
                guidance_scale=request.guidance_scale,
                model=current_model or "unknown",
            )
            
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            filename = filename_generator.generate(
                pattern=settings.get("filename_pattern", "{date}_{time}_{seed}"),
                prompt=final_prompt,
                seed=current_seed
            )
            output_path = OUTPUTS_DIR / filename
            ImageMetadata.save_with_metadata(image, output_path, metadata)
            
            images.append({
                "base64": image_to_base64(image),
                "filename": filename,
                "seed": current_seed,
                "path": f"/outputs/{filename}"
            })
        
        # 히스토리 추가
        history_manager.add(
            prompt=request.prompt,
            settings={
                "width": request.width,
                "height": request.height,
                "steps": request.steps,
                "guidance_scale": request.guidance_scale,
                "seed": seed,
            }
        )
        
        await manager.broadcast({
            "type": "complete",
            "content": f"✅ {len(images)}장 생성 완료! (시드: {seed})"
        })
        
        return {"success": True, "images": images, "seed": seed, "prompt": final_prompt}
        
    except Exception as e:
        await manager.broadcast({"type": "error", "content": f"❌ 생성 오류: {str(e)}"})
        raise HTTPException(500, str(e))
    
    finally:
        is_generating = False


@app.post("/api/preview")
async def generate_preview(request: GenerateRequest):
    """빠른 미리보기 (256x256)"""
    request.width = 256
    request.height = 256
    request.steps = min(request.steps, 4)
    request.num_images = 1
    return await generate_image(request)


@app.post("/api/translate")
async def translate_text(request: TranslateRequest):
    """프롬프트 번역 (한국어 → 영어)"""
    from utils.llm_client import llm_client
    
    if not llm_client.is_available:
        raise HTTPException(400, "LLM API가 설정되지 않았습니다. 설정에서 API 키를 입력해주세요.")
    
    translated, success = translator.translate(request.text)
    return {"success": success, "translated": translated}


@app.post("/api/translate-reverse")
async def reverse_translate_text(request: TranslateRequest):
    """프롬프트 역번역 (영어 → 한국어)"""
    from utils.llm_client import llm_client
    
    if not llm_client.is_available:
        raise HTTPException(400, "LLM API가 설정되지 않았습니다. 설정에서 API 키를 입력해주세요.")
    
    translated, success = translator.reverse_translate(request.text)
    return {"success": success, "translated": translated}


@app.post("/api/enhance")
async def enhance_prompt(request: EnhanceRequest):
    """프롬프트 향상"""
    from utils.llm_client import llm_client
    
    if not llm_client.is_available:
        raise HTTPException(400, "LLM API가 설정되지 않았습니다. 설정에서 API 키를 입력해주세요.")
    
    enhanced, success = prompt_enhancer.enhance(request.prompt, request.style)
    return {"success": success, "enhanced": enhanced}


@app.get("/api/templates")
async def get_templates():
    """프롬프트 템플릿 목록"""
    return {"templates": PROMPT_TEMPLATES}


@app.get("/api/model-status")
async def get_model_download_status():
    """각 모델의 다운로드 상태 확인"""
    from huggingface_hub import try_to_load_from_cache, scan_cache_dir
    import os
    
    status = {}
    
    for option_name, option_info in QUANTIZATION_OPTIONS.items():
        is_downloaded = False
        
        try:
            if option_info.get("is_gguf", False):
                # GGUF 모델: 특정 파일이 캐시에 있는지 확인
                filename = option_info.get("filename", "")
                repo_id = option_info.get("repo", "")
                
                if filename and repo_id:
                    cached_path = try_to_load_from_cache(
                        repo_id=repo_id,
                        filename=filename
                    )
                    is_downloaded = cached_path is not None
            else:
                # BF16 모델: diffusers 캐시 확인
                repo_id = option_info.get("repo", "")
                if repo_id:
                    # model_index.json이 있으면 다운로드된 것으로 간주
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


@app.get("/api/history")
async def get_history():
    """히스토리 목록"""
    entries = history_manager.get_all()
    return {"history": [e.to_dict() for e in entries[:50]]}


@app.delete("/api/history")
async def clear_history():
    """히스토리 삭제"""
    history_manager.clear()
    return {"success": True}


@app.get("/api/favorites")
async def get_favorites():
    """즐겨찾기 목록"""
    entries = favorites_manager.get_all()
    return {"favorites": [e.to_dict() for e in entries]}


@app.post("/api/favorites")
async def add_favorite(request: FavoriteRequest):
    """즐겨찾기 추가"""
    entry = favorites_manager.add(
        name=request.name,
        prompt=request.prompt,
        settings=request.settings
    )
    return {"success": True, "id": entry.id}


@app.delete("/api/favorites/{fav_id}")
async def delete_favorite(fav_id: str):
    """즐겨찾기 삭제"""
    success = favorites_manager.delete(fav_id)
    return {"success": success}


@app.get("/api/gallery")
async def get_gallery():
    """갤러리 이미지 목록"""
    images = []
    if OUTPUTS_DIR.exists():
        for f in sorted(OUTPUTS_DIR.glob("*.png"), key=lambda x: x.stat().st_mtime, reverse=True)[:50]:
            metadata = ImageMetadata.read_metadata(f)
            images.append({
                "filename": f.name,
                "path": f"/outputs/{f.name}",
                "metadata": metadata
            })
    return {"images": images}


@app.post("/api/settings")
async def save_settings(request: SettingsRequest):
    """설정 저장"""
    from utils.llm_client import llm_client
    
    # 레거시 호환: openai_api_key가 있고 llm_api_key가 없으면 동기화
    if request.openai_api_key:
        settings.set("openai_api_key", request.openai_api_key)
        # 레거시 호환 유지
        translator.set_api_key(request.openai_api_key)
        prompt_enhancer.set_api_key(request.openai_api_key)
    
    # LLM Provider 설정
    if request.llm_provider:
        settings.set("llm_provider", request.llm_provider)
    
    if request.llm_api_key:
        settings.set("llm_api_key", request.llm_api_key)
        # 레거시 호환 동기화
        settings.set("openai_api_key", request.llm_api_key)
    
    if request.llm_base_url is not None:
        settings.set("llm_base_url", request.llm_base_url)
    
    if request.llm_model is not None:
        settings.set("llm_model", request.llm_model)
    
    # LLM 클라이언트 캐시 무효화 (설정 변경 반영)
    llm_client.invalidate()
    
    if request.output_path:
        settings.set("output_path", request.output_path)
    
    if request.filename_pattern:
        settings.set("filename_pattern", request.filename_pattern)
    
    # 시스템 프롬프트 설정
    if request.translate_system_prompt is not None:
        settings.set("translate_system_prompt", request.translate_system_prompt)
    
    if request.enhance_system_prompt is not None:
        settings.set("enhance_system_prompt", request.enhance_system_prompt)
    
    return {"success": True}


@app.get("/api/settings")
async def get_settings():
    """설정 가져오기"""
    from utils.settings import LLM_PROVIDERS
    from utils.translator import Translator
    from utils.prompt_enhancer import PromptEnhancer
    
    return {
        # 레거시 호환
        "openai_api_key": "***" if settings.get("openai_api_key") else "",
        # LLM Provider 설정
        "llm_provider": settings.get("llm_provider", "openai"),
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
        # 시스템 프롬프트 (번역/향상)
        "translate_system_prompt": settings.get("translate_system_prompt") or Translator.DEFAULT_SYSTEM_PROMPT,
        "enhance_system_prompt": settings.get("enhance_system_prompt") or PromptEnhancer.DEFAULT_SYSTEM_PROMPT,
        "default_translate_system_prompt": Translator.DEFAULT_SYSTEM_PROMPT,
        "default_enhance_system_prompt": PromptEnhancer.DEFAULT_SYSTEM_PROMPT,
        # 기타 설정
        "output_path": str(settings.get("output_path", OUTPUTS_DIR)),
        "filename_pattern": settings.get("filename_pattern", "{date}_{time}_{seed}"),
        "quantization_options": list(QUANTIZATION_OPTIONS.keys()),
        "resolution_presets": RESOLUTION_PRESETS,
    }


# ============= WebSocket =============
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """웹소켓 연결"""
    await manager.connect(websocket)
    try:
        # 연결 시 상태 전송
        await websocket.send_json({
            "type": "connected",
            "content": "서버에 연결되었습니다."
        })
        
        while True:
            data = await websocket.receive_text()
            # 클라이언트 메시지 처리 (필요시)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ============= 메인 =============
if __name__ == "__main__":
    # 출력 폴더 생성
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("🎨 Z-Image WebUI 시작...")
    print("📍 http://localhost:7860")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        reload=False
    )
