"""Z-Image WebUI 기본 설정값"""

import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

# 기본 경로
BASE_DIR = Path(__file__).parent.parent

# .env 기본 내용 (ASCII only for encoding safety)
DEFAULT_ENV_CONTENT = """# ===== Z-Image WebUI Environment Settings =====

# ===== Server Settings =====
PORT=7860
HOST=0.0.0.0
RELOAD=false

# ===== Model Settings =====
DEFAULT_MODEL=Tongyi-MAI/Z-Image-Turbo
GGUF_MODEL_REPO=jayn7/Z-Image-Turbo-GGUF

# ===== LongCat-Image-Edit Model Settings =====
LONGCAT_EDIT_MODEL=meituan-longcat/LongCat-Image-Edit
LONGCAT_EDIT_AUTO_UNLOAD_TIMEOUT=10

# ===== LLM Settings =====
# Provider: openai, groq, openrouter, together, ollama, lmstudio, custom
LLM_PROVIDER=openai
LLM_API_KEY=
LLM_MODEL=
# LLM_BASE_URL=  # For custom provider only

# ===== Debug =====
DEBUG=false
"""

def _ensure_env_file():
    """
    .env 파일이 없으면 자동 생성
    - .env.example이 있으면 복사
    - 없으면 기본값으로 생성
    """
    env_path = BASE_DIR / ".env"
    env_example_path = BASE_DIR / ".env.example"
    
    if not env_path.exists():
        if env_example_path.exists():
            # .env.example 복사
            shutil.copy(env_example_path, env_path)
            print(f"📝 .env 파일 생성됨 (.env.example 복사)")
        else:
            # 기본값으로 생성
            env_path.write_text(DEFAULT_ENV_CONTENT, encoding="utf-8")
            print(f"📝 .env 파일 생성됨 (기본값)")

# .env 파일 확인 및 생성
_ensure_env_file()

# .env 파일 로드
load_dotenv(BASE_DIR / ".env")

# ===== 서버 설정 (환경변수에서 로드) =====
SERVER_HOST = os.getenv("HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("PORT", "7860"))
SERVER_RELOAD = os.getenv("RELOAD", "false").lower() == "true"
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# ===== 기본 경로 =====
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", str(BASE_DIR / "outputs")))
MODELS_DIR = Path.home() / ".cache" / "huggingface" / "hub"

# ===== 모델 설정 =====
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "Tongyi-MAI/Z-Image-Turbo")
GGUF_MODEL_REPO = os.getenv("GGUF_MODEL_REPO", "jayn7/Z-Image-Turbo-GGUF")

# ===== LongCat-Image-Edit 모델 설정 =====
LONGCAT_EDIT_MODEL = os.getenv("LONGCAT_EDIT_MODEL", "meituan-longcat/LongCat-Image-Edit")
LONGCAT_EDIT_AUTO_UNLOAD_TIMEOUT = int(os.getenv("LONGCAT_EDIT_AUTO_UNLOAD_TIMEOUT", "10"))

# ===== LLM 설정 (환경변수 우선) =====
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "")  # 빈 문자열이면 settings.yaml 사용
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")

# ===== GPU 설정 (관리자 전용) =====
# 사용 가능한 값: "auto", "cuda:0", "cuda:1", ..., "cpu", "mps"
DEFAULT_GPU_SETTINGS = {
    "generation_gpu": os.getenv("GENERATION_GPU", "auto"),  # 이미지 생성 모델 GPU
    "edit_gpu": os.getenv("EDIT_GPU", "auto"),              # 이미지 편집 모델 GPU
}

# 모델 옵션 (BF16 전용 및 GGUF 양자화 옵션)
QUANTIZATION_OPTIONS = {
    # 기본 BF16 (양자화 없음, 최고 품질)
    "BF16 (기본, 최고품질)": {
        "type": "bf16", 
        "repo": DEFAULT_MODEL,
        "is_gguf": False,
    },
    # GGUF 양자화 옵션 (VRAM 절약)
    "GGUF Q8_0 (7.22GB, 고품질)": {
        "type": "Q8_0",
        "repo": GGUF_MODEL_REPO,
        "filename": "z_image_turbo-Q8_0.gguf",
        "is_gguf": True,
    },
    "GGUF Q6_K (5.91GB, 고품질)": {
        "type": "Q6_K",
        "repo": GGUF_MODEL_REPO,
        "filename": "z_image_turbo-Q6_K.gguf",
        "is_gguf": True,
    },
    "GGUF Q5_K_M (5.52GB, 균형)": {
        "type": "Q5_K_M",
        "repo": GGUF_MODEL_REPO,
        "filename": "z_image_turbo-Q5_K_M.gguf",
        "is_gguf": True,
    },
    "GGUF Q5_K_S (5.19GB, 균형)": {
        "type": "Q5_K_S",
        "repo": GGUF_MODEL_REPO,
        "filename": "z_image_turbo-Q5_K_S.gguf",
        "is_gguf": True,
    },
    "GGUF Q4_K_M (4.98GB, 추천)": {
        "type": "Q4_K_M",
        "repo": GGUF_MODEL_REPO,
        "filename": "z_image_turbo-Q4_K_M.gguf",
        "is_gguf": True,
    },
    "GGUF Q4_K_S (4.66GB, 경량)": {
        "type": "Q4_K_S",
        "repo": GGUF_MODEL_REPO,
        "filename": "z_image_turbo-Q4_K_S.gguf",
        "is_gguf": True,
    },
    "GGUF Q3_K_M (4.12GB, 저사양)": {
        "type": "Q3_K_M",
        "repo": GGUF_MODEL_REPO,
        "filename": "z_image_turbo-Q3_K_M.gguf",
        "is_gguf": True,
    },
    "GGUF Q3_K_S (3.79GB, 최저사양)": {
        "type": "Q3_K_S",
        "repo": GGUF_MODEL_REPO,
        "filename": "z_image_turbo-Q3_K_S.gguf",
        "is_gguf": True,
    },
}

# 이미지 생성 기본값
DEFAULT_GENERATION_SETTINGS = {
    "width": 512,
    "height": 512,
    "num_inference_steps": 9,
    "guidance_scale": 0.0,
    "num_images": 1,
    "seed": -1,  # -1 = 랜덤
}

# LongCat-Image-Edit 모델 옵션
# 양자화 옵션: bf16 (기본), int8 (bitsandbytes 8bit), int4 (bitsandbytes 4bit)
EDIT_QUANTIZATION_OPTIONS = {
    "BF16 (기본, 최고품질)": {
        "type": "bf16",
        "repo": LONGCAT_EDIT_MODEL,
        "quantization": None,
        "estimated_vram": "~20-24GB",
    },
    "INT8 (절반 용량, 고품질)": {
        "type": "int8",
        "repo": LONGCAT_EDIT_MODEL,
        "quantization": "int8",
        "estimated_vram": "~12-14GB",
    },
    "INT4 (1/4 용량, 균형)": {
        "type": "int4",
        "repo": LONGCAT_EDIT_MODEL,
        "quantization": "int4",
        "estimated_vram": "~8-10GB",
    },
}

# 이미지 편집 기본값
DEFAULT_EDIT_SETTINGS = {
    "num_inference_steps": 50,
    "guidance_scale": 4.5,
    "num_images": 1,
    "seed": -1,  # -1 = 랜덤
}

# 해상도 프리셋
RESOLUTION_PRESETS = {
    "512x512 (정사각형)": (512, 512),
    "768x768 (정사각형 HD)": (768, 768),
    "1024x1024 (정사각형 Full HD)": (1024, 1024),
    "512x768 (세로)": (512, 768),
    "768x512 (가로)": (768, 512),
    "768x1024 (세로 HD)": (768, 1024),
    "1024x768 (가로 HD)": (1024, 768),
    "커스텀": None,
}

# 스텝 프리셋
STEP_PRESETS = {
    "빠름 (4 steps)": 4,
    "기본 (8 steps)": 8,
    "품질 (12 steps)": 12,
    "고품질 (16 steps)": 16,
    "최고품질 (20 steps)": 20,
}

# 파일명 패턴
FILENAME_PATTERNS = {
    "{date}_{time}_{seed}": "날짜_시간_시드",
    "{prompt_short}_{seed}": "프롬프트_시드",
    "{date}_{prompt_short}_{seed}": "날짜_프롬프트_시드",
    "image_{counter}_{seed}": "image_번호_시드",
}

# UI 테마
THEMES = {
    "다크 모드": "dark",
    "라이트 모드": "light",
    "시스템 설정": "system",
}
