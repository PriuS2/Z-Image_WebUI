"""Z-Image WebUI 기본 설정값"""

import os
from pathlib import Path

# 기본 경로
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = Path.home() / ".cache" / "huggingface" / "hub"

# 기본 모델 설정
DEFAULT_MODEL = "Tongyi-MAI/Z-Image-Turbo"

# GGUF 모델 저장소
GGUF_MODEL_REPO = "jayn7/Z-Image-Turbo-GGUF"

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

# 업스케일 옵션
UPSCALE_OPTIONS = {
    "2x": 2,
    "4x": 4,
}

# UI 테마
THEMES = {
    "다크 모드": "dark",
    "라이트 모드": "light",
    "시스템 설정": "system",
}

# ============= ControlNet 설정 =============

# ControlNet 모델 저장소
CONTROLNET_MODEL_REPO = "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union"
CONTROLNET_MODEL_FILENAME = "Z-Image-Turbo-Fun-Controlnet-Union.safetensors"

# 컨트롤 타입 정의
CONTROL_TYPES = {
    "canny": {
        "name": "Canny Edge",
        "description": "윤곽선 추출 (날카로운 엣지)",
        "icon": "ri-pencil-ruler-2-line",
    },
    "depth": {
        "name": "Depth Map",
        "description": "깊이 맵 추출 (원근감)",
        "icon": "ri-3d-cude-sphere-line",
    },
    "pose": {
        "name": "Pose",
        "description": "인체 포즈 스켈레톤 추출",
        "icon": "ri-body-scan-line",
    },
    "hed": {
        "name": "HED Edge",
        "description": "부드러운 윤곽선 추출",
        "icon": "ri-brush-line",
    },
    "mlsd": {
        "name": "MLSD Lines",
        "description": "직선 구조 추출 (건축물)",
        "icon": "ri-layout-grid-line",
    },
}

# ControlNet 기본 설정
DEFAULT_CONTROLNET_SETTINGS = {
    "control_context_scale": 0.7,  # 권장 범위: 0.65 ~ 0.80
    "control_context_scale_min": 0.5,
    "control_context_scale_max": 1.0,
}

# Canny 전처리 기본값
CANNY_DEFAULTS = {
    "low_threshold": 100,
    "high_threshold": 200,
}

# MLSD 전처리 기본값
MLSD_DEFAULTS = {
    "thr_v": 0.1,
    "thr_d": 0.1,
}

# Pose 전처리 기본값
POSE_DEFAULTS = {
    "include_hand": False,
    "include_face": False,
}