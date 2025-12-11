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

# GPU 설정 (다중 GPU 지원)
GPU_OPTIONS = {
    "auto": "자동 선택 (가장 여유로운 GPU)",
    "cuda:0": "GPU 0",
    "cuda:1": "GPU 1",
    "cuda:2": "GPU 2",
    "cuda:3": "GPU 3",
    "multi": "다중 GPU (모델 분산)",
}

# 메모리 최적화 옵션
MEMORY_OPTIMIZATION = {
    "none": "없음 (최대 성능)",
    "attention_slicing": "Attention Slicing (중간 메모리 절약)",
    "vae_tiling": "VAE Tiling (고해상도 지원)",
    "sequential_cpu_offload": "Sequential CPU Offload (최대 메모리 절약)",
    "model_cpu_offload": "Model CPU Offload (균형)",
    "all": "모든 최적화 적용",
}