# Qwen-Image-Edit-2511 완벽 가이드

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Qwen--Image--Edit--2511-blue)](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2508.02324-b31b1b.svg)](https://arxiv.org/abs/2508.02324)

> **Qwen-Image-Edit-2511**은 Alibaba의 Qwen 팀에서 개발한 고급 이미지 편집 AI 모델로, 이전 버전인 Qwen-Image-Edit-2509를 크게 개선한 버전입니다.

> 📌 **이 문서는 Windows 환경을 기준으로 작성되었습니다.**

---

## 📋 목차

- [개요](#개요)
- [주요 개선 사항](#주요-개선-사항)
- [모델 아키텍처](#모델-아키텍처)
- [시스템 요구 사항](#시스템-요구-사항)
- [설치 방법](#설치-방법)
- [빠른 시작](#빠른-시작)
- [파라미터 상세 가이드](#파라미터-상세-가이드)
- [다양한 사용 예제](#다양한-사용-예제)
- [최적화 방법](#최적화-방법)
- [LoRA 활용 가이드](#lora-활용-가이드)
- [문제 해결](#문제-해결)
- [라이선스 및 인용](#라이선스-및-인용)
- [참고 자료](#참고-자료)

---

## 개요

**Qwen-Image-Edit-2511**은 텍스트 프롬프트를 기반으로 이미지를 편집하는 최첨단 AI 모델입니다. Diffusion Transformer 아키텍처를 기반으로 하며, 약 **20B(200억)** 개의 파라미터를 보유하고 있습니다.

### 핵심 기능

| 기능 | 설명 |
|------|------|
| **시멘틱 편집** | 객체 추가, 제거, 수정, 회전, 스타일 변환 |
| **정확한 텍스트 편집** | 이미지 내 텍스트 추가/삭제/수정 (영어, 중국어 지원) |
| **다중 이미지 입력** | 최대 3개의 입력 이미지를 활용한 복합 편집 |
| **캐릭터 일관성** | 인물의 정체성과 특징을 유지하면서 편집 |
| **LoRA 통합** | 별도 튜닝 없이 다양한 스타일 적용 가능 |
| **산업 디자인** | 제품 디자인 및 마케팅 비주얼 제작 지원 |

---

## 주요 개선 사항

Qwen-Image-Edit-2511은 이전 버전(2509) 대비 다음과 같은 주요 개선이 이루어졌습니다:

### 1. 🎯 이미지 드리프트 완화 (Mitigate Image Drift)

- 반복적인 편집 과정에서 원본 이미지의 정체성이 흐려지는 문제를 해결
- 전체적인 이미지 구조와 대상의 특징이 안정적으로 유지
- 여러 차례 수정을 거쳐도 이미지 품질 저하 최소화

### 2. 👥 캐릭터 일관성 향상 (Improved Character Consistency)

- 단일 인물 편집 시 인물의 정체성과 시각적 특징 보존
- **다중 인물 사진**에서도 각 인물의 일관성 유지
- 두 개의 별도 인물 이미지를 자연스러운 그룹 사진으로 합성 가능

### 3. 🎨 LoRA 기능 통합 (Integrated LoRA Capabilities)

커뮤니티에서 개발된 인기 있는 LoRA들이 기본 모델에 통합되어 있습니다:

- **조명 향상 LoRA**: 현실적인 조명 제어
- **뷰포인트 생성**: 새로운 시점에서의 이미지 생성
- 추가적인 튜닝 없이 바로 사용 가능

### 4. 🏭 산업 디자인 생성 강화 (Enhanced Industrial Design)

- 산업 제품 배치 디자인
- 산업 부품의 재료 교체
- 반복적인 디자인 시안 생성
- 마케팅 비주얼 제작

### 5. 📐 기하학적 추론 능력 강화 (Strengthened Geometric Reasoning)

- 설계나 주석을 위한 보조 구성선(auxiliary construction lines) 직접 생성
- 구조적 변형 및 형태 인식 편집의 정확도 향상
- 기하학적 구조 이해 능력 개선

---

## 모델 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                    Qwen-Image-Edit-2511                         │
├─────────────────────────────────────────────────────────────────┤
│  Base Architecture: Diffusion Transformer (DiT)                 │
│  Parameters: ~20B                                               │
│  Input: Text Prompt + Up to 3 Images                            │
│  Output: Edited Image                                           │
│  Pipeline: QwenImageEditPlusPipeline                            │
│  Precision: BFloat16 / Float16                                  │
├─────────────────────────────────────────────────────────────────┤
│  Components:                                                    │
│  ├── Text Encoder (Multilingual: EN, ZH)                        │
│  ├── Vision Encoder                                             │
│  ├── Diffusion Transformer                                      │
│  └── VAE Decoder                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 시스템 요구 사항

### 최소 요구 사항

| 구성 요소 | 최소 사양 | 권장 사양 |
|-----------|-----------|-----------|
| **GPU** | NVIDIA RTX 3090 (24GB) | NVIDIA RTX 4090 / A100 |
| **VRAM** | 24GB | 40GB+ |
| **RAM** | 32GB | 64GB |
| **저장 공간** | 50GB | 100GB |
| **Python** | 3.10+ | 3.11+ |
| **CUDA** | 11.8+ | 12.1+ |

### 권장 환경 (Windows)

```powershell
# Python 버전 확인
python --version  # 3.10 이상 권장

# CUDA 버전 확인
nvcc --version    # 11.8 이상 권장

# GPU 메모리 확인
nvidia-smi
```

---

## 설치 방법 (Windows)

### 1. 기본 설치

```powershell
# 가상 환경 생성 (권장)
python -m venv venv

# 가상 환경 활성화 (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# 또는 명령 프롬프트(CMD)에서
# qwen-edit-env\Scripts\activate.bat

# PyTorch 설치 (CUDA 버전에 맞게)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# diffusers 최신 버전 설치 (필수)
pip install git+https://github.com/huggingface/diffusers

# 기타 필수 패키지
pip install transformers accelerate pillow
```

> 💡 **PowerShell 실행 정책 오류 시**: 
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` 실행 후 다시 시도

### 2. 전체 의존성 설치

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install git+https://github.com/huggingface/diffusers
pip install transformers>=4.40.0
pip install accelerate>=0.26.0
pip install safetensors
pip install pillow
pip install numpy
```

### requirements.txt

```txt
torch torchvision --index-url https://download.pytorch.org/whl/cu126
diffusers @ git+https://github.com/huggingface/diffusers
transformers>=4.40.0
accelerate>=0.26.0
safetensors>=0.4.0
pillow>=10.0.0
numpy>=1.24.0
```

---

## 빠른 시작

### 기본 사용법

```python
import os
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

# ═══════════════════════════════════════════════════════════════
# 1. 파이프라인 로드
# ═══════════════════════════════════════════════════════════════
pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    torch_dtype=torch.bfloat16
)
print("✅ 파이프라인 로드 완료")

# GPU로 이동
pipeline.to('cuda')
pipeline.set_progress_bar_config(disable=None)

# ═══════════════════════════════════════════════════════════════
# 2. 입력 이미지 로드
# ═══════════════════════════════════════════════════════════════
image = Image.open("input_image.png")

# ═══════════════════════════════════════════════════════════════
# 3. 프롬프트 설정 및 편집 수행
# ═══════════════════════════════════════════════════════════════
prompt = "Change the background to a sunset sky"

inputs = {
    "image": [image],
    "prompt": prompt,
    "generator": torch.manual_seed(42),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}

# ═══════════════════════════════════════════════════════════════
# 4. 이미지 생성 및 저장
# ═══════════════════════════════════════════════════════════════
with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("edited_image.png")
    print(f"✅ 이미지 저장 완료: {os.path.abspath('edited_image.png')}")
```

---

## 파라미터 상세 가이드

### 주요 파라미터

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `image` | List[PIL.Image] | 필수 | 입력 이미지 리스트 (최대 3개) |
| `prompt` | str | 필수 | 편집 지시 텍스트 |
| `negative_prompt` | str | " " | 생성에서 제외할 요소 |
| `num_inference_steps` | int | 40 | 디노이징 스텝 수 |
| `guidance_scale` | float | 1.0 | 프롬프트 가이던스 강도 |
| `true_cfg_scale` | float | 4.0 | True CFG 스케일 |
| `generator` | torch.Generator | None | 시드 제어용 생성기 |
| `num_images_per_prompt` | int | 1 | 프롬프트당 생성할 이미지 수 |

### 파라미터 상세 설명

#### `num_inference_steps` (추론 스텝 수)

```python
# 빠른 생성 (품질 낮음)
inputs["num_inference_steps"] = 20

# 균형 잡힌 설정 (권장)
inputs["num_inference_steps"] = 40

# 고품질 생성 (시간 오래 걸림)
inputs["num_inference_steps"] = 60
```

| 스텝 수 | 품질 | 속도 | 용도 |
|---------|------|------|------|
| 20-25 | 낮음 | 빠름 | 테스트/프리뷰 |
| 35-45 | 중간 | 보통 | 일반 사용 |
| 50-60+ | 높음 | 느림 | 최종 결과물 |

#### `true_cfg_scale` (True CFG 스케일)

프롬프트에 대한 모델의 충실도를 조절합니다:

```python
# 낮은 값: 더 자연스럽지만 프롬프트 반영 약함
inputs["true_cfg_scale"] = 2.0

# 권장 값
inputs["true_cfg_scale"] = 4.0

# 높은 값: 프롬프트 강하게 반영, 때때로 부자연스러움
inputs["true_cfg_scale"] = 6.0
```

#### `negative_prompt` (네거티브 프롬프트)

생성에서 제외하고 싶은 요소를 지정합니다:

```python
inputs["negative_prompt"] = "blurry, low quality, distorted, ugly, bad anatomy"
```

---

## 다양한 사용 예제

### 예제 1: 단일 이미지 편집

```python
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

# 파이프라인 로드
pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    torch_dtype=torch.bfloat16
)
pipeline.to('cuda')

# 단일 이미지 편집
image = Image.open("portrait.png")
prompt = "Add a wizard hat and cloak to the person in this photo"

inputs = {
    "image": [image],
    "prompt": prompt,
    "generator": torch.manual_seed(42),
    "true_cfg_scale": 4.0,
    "negative_prompt": "blurry, distorted",
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}

with torch.inference_mode():
    output = pipeline(**inputs)
    output.images[0].save("wizard_portrait.png")
```

### 예제 2: 두 이미지 합성 (다중 인물)

```python
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    torch_dtype=torch.bfloat16
)
pipeline.to('cuda')

# 두 개의 인물 이미지 로드
person1 = Image.open("person1.png")
person2 = Image.open("person2.png")

# 두 인물을 하나의 장면에 합성
prompt = "The person on the left and the person on the right are sitting across from each other in a coffee shop, having a conversation"

inputs = {
    "image": [person1, person2],
    "prompt": prompt,
    "generator": torch.manual_seed(123),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}

with torch.inference_mode():
    output = pipeline(**inputs)
    output.images[0].save("coffee_shop_scene.png")
```

### 예제 3: 스타일 변환 (지브리 스타일)

```python
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    torch_dtype=torch.bfloat16
)
pipeline.to('cuda')

image = Image.open("landscape.png")
prompt = "Transform this landscape photo into Studio Ghibli animation style"

inputs = {
    "image": [image],
    "prompt": prompt,
    "generator": torch.manual_seed(777),
    "true_cfg_scale": 5.0,
    "negative_prompt": "realistic, photograph",
    "num_inference_steps": 50,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}

with torch.inference_mode():
    output = pipeline(**inputs)
    output.images[0].save("ghibli_landscape.png")
```

### 예제 4: 텍스트 편집

```python
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    torch_dtype=torch.bfloat16
)
pipeline.to('cuda')

# 텍스트가 포함된 이미지
image = Image.open("poster.png")
prompt = "Change the poster title text from 'SUMMER SALE' to 'WINTER COLLECTION'"

inputs = {
    "image": [image],
    "prompt": prompt,
    "generator": torch.manual_seed(42),
    "true_cfg_scale": 4.5,
    "negative_prompt": " ",
    "num_inference_steps": 45,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}

with torch.inference_mode():
    output = pipeline(**inputs)
    output.images[0].save("updated_poster.png")
```

### 예제 5: 산업 디자인 - 재료 변경

```python
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    torch_dtype=torch.bfloat16
)
pipeline.to('cuda')

# 제품 이미지
image = Image.open("product.png")
prompt = "Change this plastic product to wood material. Apply natural wood grain texture."

inputs = {
    "image": [image],
    "prompt": prompt,
    "generator": torch.manual_seed(99),
    "true_cfg_scale": 4.0,
    "negative_prompt": "plastic, shiny, artificial",
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}

with torch.inference_mode():
    output = pipeline(**inputs)
    output.images[0].save("wooden_product.png")
```

### 예제 6: 객체 추가

```python
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    torch_dtype=torch.bfloat16
)
pipeline.to('cuda')

image = Image.open("room.png")
prompt = "Add a decorated Christmas tree in the center of the room. The tree has lights and ornaments."

inputs = {
    "image": [image],
    "prompt": prompt,
    "generator": torch.manual_seed(1225),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}

with torch.inference_mode():
    output = pipeline(**inputs)
    output.images[0].save("room_with_tree.png")
```

### 예제 7: 객체 제거

```python
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    torch_dtype=torch.bfloat16
)
pipeline.to('cuda')

image = Image.open("beach.png")
prompt = "Remove all people from the beach photo and leave only the clean beach"

inputs = {
    "image": [image],
    "prompt": prompt,
    "generator": torch.manual_seed(42),
    "true_cfg_scale": 4.0,
    "negative_prompt": "people, person, crowd",
    "num_inference_steps": 45,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}

with torch.inference_mode():
    output = pipeline(**inputs)
    output.images[0].save("empty_beach.png")
```

### 예제 8: 배치 처리 (여러 이미지 생성)

```python
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    torch_dtype=torch.bfloat16
)
pipeline.to('cuda')

image = Image.open("character.png")
prompt = "Place this character in various seasonal backgrounds"

# 여러 시드로 다양한 결과 생성
seeds = [42, 123, 456, 789]

for i, seed in enumerate(seeds):
    inputs = {
        "image": [image],
        "prompt": prompt,
        "generator": torch.manual_seed(seed),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 40,
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
    }
    
    with torch.inference_mode():
        output = pipeline(**inputs)
        output.images[0].save(f"character_season_{i+1}.png")
        print(f"✅ 이미지 {i+1} 저장 완료")
```

---

## 최적화 방법

### 1. 🚀 메모리 최적화

#### CPU 오프로딩

VRAM이 부족한 경우 사용합니다:

```python
from diffusers import QwenImageEditPlusPipeline
import torch

# 모델 로드
pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    torch_dtype=torch.bfloat16
)

# 방법 1: Model CPU Offload (권장)
# 사용하지 않는 모델 컴포넌트를 CPU로 이동
pipeline.enable_model_cpu_offload()

# 방법 2: Sequential CPU Offload
# 더 적은 VRAM 사용, 하지만 더 느림
pipeline.enable_sequential_cpu_offload()
```

#### Attention Slicing

메모리 사용량을 줄이는 또 다른 방법:

```python
# Attention 슬라이싱 활성화
pipeline.enable_attention_slicing("max")

# 또는 자동 설정
pipeline.enable_attention_slicing("auto")
```

#### VAE 슬라이싱

```python
# VAE 슬라이싱 (메모리 절약)
pipeline.enable_vae_slicing()

# VAE 타일링 (대용량 이미지)
pipeline.enable_vae_tiling()
```

### 2. ⚡ 속도 최적화

#### torch.compile (PyTorch 2.0+)

> ⚠️ **Windows 주의사항**: `torch.compile`은 Windows에서 **제한적으로 지원**됩니다.
> - Triton 백엔드가 Windows를 지원하지 않아 `mode="reduce-overhead"` 사용 불가
> - `mode="default"` 또는 `mode="max-autotune"` 사용 시에도 일부 기능 제한
> - Linux 환경에서만 완전한 성능 향상을 기대할 수 있음

```python
import torch

# Windows에서는 제한적으로 동작
# Linux에서 최적의 성능 발휘
try:
    pipeline.transformer = torch.compile(
        pipeline.transformer,
        mode="default",  # Windows에서는 "default" 권장
    )
    print("✅ torch.compile 적용 성공")
except Exception as e:
    print(f"⚠️ torch.compile 실패 (Windows에서 정상): {e}")
```

#### xFormers 사용

> ⚠️ **Windows 설치 시 주의사항**: xFormers는 Windows에서 설치가 까다로울 수 있습니다.
> PyTorch 버전과 CUDA 버전이 정확히 일치해야 합니다.

```powershell
# Windows에서 xFormers 설치
# 방법 1: pip로 직접 설치 (PyTorch/CUDA 버전 일치 필요)
pip install xformers

# 방법 2: 특정 버전 명시 설치 (권장)
# PyTorch 2.5.1 + CUDA 12.4 기준
pip install xformers==0.0.28.post3
```

```python
# xFormers memory efficient attention 활성화
try:
    pipeline.enable_xformers_memory_efficient_attention()
    print("✅ xFormers 활성화 성공")
except Exception as e:
    print(f"⚠️ xFormers 사용 불가: {e}")
    print("   → Attention Slicing으로 대체합니다.")
    pipeline.enable_attention_slicing("auto")
```

#### ~~Flash Attention 2~~ (Windows 미지원)

> ❌ **Windows에서 사용 불가**: Flash Attention 2는 **Linux 전용**입니다.
> - Windows에서는 공식 지원되지 않음
> - WSL2(Windows Subsystem for Linux) 환경에서는 사용 가능
> - Windows에서는 **xFormers** 또는 **Attention Slicing**을 대안으로 사용

```python
# ❌ Windows에서는 아래 코드가 동작하지 않습니다
# from diffusers import QwenImageEditPlusPipeline
# import torch
# 
# pipeline = QwenImageEditPlusPipeline.from_pretrained(
#     "Qwen/Qwen-Image-Edit-2511",
#     torch_dtype=torch.bfloat16,
#     use_flash_attention_2=True  # Linux 전용
# )

# ✅ Windows 대안: xFormers 또는 Attention Slicing 사용
pipeline.enable_attention_slicing("auto")
```

### 3. 🔢 양자화 (Quantization)

#### 4-bit 양자화 (bitsandbytes)

> ⚠️ **Windows 주의사항**: `bitsandbytes`는 공식적으로 **Linux 전용**입니다.
> - Windows에서는 비공식 빌드(`bitsandbytes-windows`)를 사용해야 함
> - 안정성이 보장되지 않으므로 **커뮤니티 양자화 모델 사용을 권장**

```powershell
# Windows에서 bitsandbytes 설치 (비공식)
pip install bitsandbytes-windows

# 또는 최신 bitsandbytes (Windows 실험적 지원)
pip install bitsandbytes>=0.43.0
```

```python
from diffusers import QwenImageEditPlusPipeline
import torch

# ⚠️ Windows에서는 불안정할 수 있음
try:
    from transformers import BitsAndBytesConfig
    
    # 4-bit 양자화 설정
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # 양자화된 모델 로드
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2511",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16
    )
    print("✅ 4-bit 양자화 로드 성공")
except Exception as e:
    print(f"❌ 양자화 실패: {e}")
    print("   → 커뮤니티 양자화 모델을 사용하세요.")
```

#### 커뮤니티 양자화 모델 사용 (Windows 권장 ✅)

Windows 환경에서는 이미 양자화된 커뮤니티 모델을 사용하는 것이 가장 안정적입니다.

##### ovedrive/Qwen-Image-Edit-2511-4bit (권장)

[ovedrive/Qwen-Image-Edit-2511-4bit](https://huggingface.co/ovedrive/Qwen-Image-Edit-2511-4bit)는 **NF4 양자화** 모델로, **20GB 미만의 VRAM**에서 실행 가능합니다.

**특징:**
- ✅ diffusers에서 바로 사용 가능
- ✅ 16GB VRAM에서도 실행 가능
- ✅ 중요 레이어는 full precision 유지 (품질 보장)
- ✅ 10 스텝에서도 작동

**VRAM 요구량:**

| 설정 | VRAM 요구량 | 설명 |
|------|-------------|------|
| `pipeline.to("cuda")` | ~20GB | 전체 GPU 로드 |
| `enable_model_cpu_offload()` | ~16GB | CPU 오프로딩 |

##### 사용 예제 코드

```python
import os
from PIL import Image
import torch
from diffusers import QwenImageEditPlusPipeline

# ═══════════════════════════════════════════════════════════════
# NF4 양자화 모델 로드 (20GB 미만 VRAM에서 실행 가능)
# ═══════════════════════════════════════════════════════════════
model_path = "ovedrive/Qwen-Image-Edit-2511-4bit"
pipeline = QwenImageEditPlusPipeline.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16
)
print("✅ Pipeline loaded")

# VRAM 설정
# 방법 1: 20GB+ VRAM이 있는 경우
# pipeline.to("cuda")

# 방법 2: 16GB VRAM (권장)
pipeline.enable_model_cpu_offload()

pipeline.set_progress_bar_config(disable=None)

# ═══════════════════════════════════════════════════════════════
# 이미지 편집
# ═══════════════════════════════════════════════════════════════
image = Image.open("./input.png").convert("RGB")
prompt = "Change the background to a sunset sky"

inputs = {
    "image": image,
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 20,  # 10 스텝도 가능
}

with torch.inference_mode():
    output = pipeline(**inputs)

output_image = output.images[0]
output_image.save("output_image.png")
print(f"✅ Image saved at {os.path.abspath('output_image.png')}")
```

##### 다중 이미지 입력 예제

```python
import os
from PIL import Image
import torch
from diffusers import QwenImageEditPlusPipeline

# 모델 로드
pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "ovedrive/Qwen-Image-Edit-2511-4bit", 
    torch_dtype=torch.bfloat16
)
pipeline.enable_model_cpu_offload()

# 두 개의 이미지 로드
image1 = Image.open("person1.png").convert("RGB")
image2 = Image.open("person2.png").convert("RGB")

prompt = "The person on the left and person on the right are sitting together in a coffee shop"

inputs = {
    "image": [image1, image2],
    "prompt": prompt,
    "generator": torch.manual_seed(42),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 20,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}

with torch.inference_mode():
    output = pipeline(**inputs)
    output.images[0].save("combined_scene.png")
```

> 📖 모델 페이지: [ovedrive/Qwen-Image-Edit-2511-4bit](https://huggingface.co/ovedrive/Qwen-Image-Edit-2511-4bit)

### 4. 📊 추론 스텝 최적화

#### 스텝 디스틸레이션

빠른 추론을 위한 Lightning/Fast 버전 사용:

```python
# Fast 버전 (커뮤니티 스페이스에서 사용 가능)
# https://huggingface.co/spaces/linoyts/Qwen-Image-Edit-2511-Fast

# 더 적은 스텝으로 빠른 결과
inputs["num_inference_steps"] = 8  # Lightning 버전
# 또는
inputs["num_inference_steps"] = 4  # Turbo 버전
```

### 5. 🎯 종합 최적화 예제 (Windows)

```python
import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image

def create_optimized_pipeline_windows():
    """Windows 환경에 최적화된 파이프라인 생성"""
    
    # 1. 기본 파이프라인 로드
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2511",
        torch_dtype=torch.bfloat16,
    )
    
    # 2. GPU로 이동
    pipeline.to('cuda')
    
    # 3. 메모리 최적화 (Windows 호환)
    pipeline.enable_attention_slicing("auto")
    pipeline.enable_vae_slicing()
    
    # 4. xFormers 사용 시도 (설치된 경우)
    try:
        pipeline.enable_xformers_memory_efficient_attention()
        print("✅ xFormers 활성화")
    except Exception as e:
        print(f"⚠️ xFormers 사용 불가: {e}")
        print("   → Attention Slicing으로 대체됨")
    
    # ❌ Windows에서 사용 불가:
    # - Flash Attention 2
    # - torch.compile (제한적)
    
    return pipeline

def edit_image_optimized(pipeline, image_path, prompt, seed=42):
    """최적화된 이미지 편집"""
    
    image = Image.open(image_path)
    
    inputs = {
        "image": [image],
        "prompt": prompt,
        "generator": torch.manual_seed(seed),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 30,  # 품질과 속도의 균형
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
    }
    
    with torch.inference_mode():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output = pipeline(**inputs)
    
    return output.images[0]

# 사용 예시
if __name__ == "__main__":
    pipeline = create_optimized_pipeline_windows()
    result = edit_image_optimized(
        pipeline,
        "input.png",
        "Change the background to outer space"
    )
    result.save("output.png")
    print("✅ 이미지 저장 완료!")
```

### 메모리 사용량 비교표 (Windows)

| 최적화 방법 | VRAM 사용량 | 속도 영향 | 품질 영향 | Windows 지원 |
|-------------|-------------|-----------|-----------|--------------|
| 기본 (BF16) | ~24GB | 기준 | 기준 | ✅ |
| CPU Offload | ~16GB | -30% | 없음 | ✅ |
| Sequential Offload | ~8GB | -60% | 없음 | ✅ |
| Attention Slicing | ~20GB | -10% | 없음 | ✅ |
| VAE Slicing | ~22GB | -5% | 없음 | ✅ |
| VAE Tiling | ~20GB | -15% | 없음 | ✅ |
| NF4 양자화 (ovedrive) | ~16GB | -10% | 약간 저하 | ✅ |
| xFormers | ~22GB | +20% | 없음 | ⚠️ 설치 주의 |
| Flash Attention 2 | ~20GB | +30% | 없음 | ❌ 미지원 |
| torch.compile | - | +15~40% | 없음 | ⚠️ 제한적 |

> 💡 **Windows 권장 조합**: CPU Offload + Attention Slicing + VAE Slicing

---

## LoRA 활용 가이드

### 내장 LoRA 기능

Qwen-Image-Edit-2511에는 인기 있는 커뮤니티 LoRA가 기본 통합되어 있습니다:

#### 조명 향상 (Lighting Enhancement)

```python
# 조명 관련 프롬프트를 사용하면 자동으로 적용됨
prompt = "Add dramatic golden lighting to this photo"
prompt = "Add dramatic rim lighting to the subject"
prompt = "Studio lighting setup with soft boxes"
```

#### 뷰포인트 변경 (Viewpoint Generation)

```python
# 다른 각도에서 본 이미지 생성
prompt = "Show this same scene from a bird's eye view"
prompt = "Show this scene from a bird's eye view"
prompt = "Render this object from a 45-degree angle"
```

### 커스텀 LoRA 적용

```python
from diffusers import QwenImageEditPlusPipeline
import torch

pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    torch_dtype=torch.bfloat16
)
pipeline.to('cuda')

# LoRA 가중치 로드
pipeline.load_lora_weights(
    "path/to/lora",
    weight_name="custom_lora.safetensors"
)

# LoRA 스케일 조정 (0.0 ~ 1.0)
pipeline.fuse_lora(lora_scale=0.8)

# 이미지 편집 수행
# ...
```

### 인기 있는 커뮤니티 LoRA 목록

| LoRA 이름 | 용도 | 출처 |
|-----------|------|------|
| Lighting Enhancement | 조명 제어 | 내장 |
| Viewpoint Generation | 시점 변경 | 내장 |
| Anime Style | 애니메이션 스타일 | CivitAI |
| Realistic Skin | 피부 질감 개선 | CivitAI |
| Product Photography | 제품 사진 | HuggingFace |

---

## 문제 해결

### 자주 발생하는 오류 및 해결 방법

#### 1. CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**해결 방법:**

```python
# 방법 1: CPU 오프로딩
pipeline.enable_model_cpu_offload()

# 방법 2: 메모리 정리
import torch
torch.cuda.empty_cache()

# 방법 3: 더 작은 이미지 사용
from PIL import Image
image = Image.open("large_image.png")
image = image.resize((512, 512))
```

#### 2. 모델 로드 실패

```
OSError: Can't load the model
```

**해결 방법:**

```powershell
# diffusers 최신 버전 설치
pip install --upgrade git+https://github.com/huggingface/diffusers

# Windows에서 캐시 정리 (PowerShell)
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface\hub\models--Qwen--Qwen-Image-Edit-2511"

# 또는 명령 프롬프트(CMD)에서
rmdir /s /q "%USERPROFILE%\.cache\huggingface\hub\models--Qwen--Qwen-Image-Edit-2511"
```

#### 3. 프롬프트가 제대로 반영되지 않음

**해결 방법:**

```python
# true_cfg_scale 값 조정
inputs["true_cfg_scale"] = 5.0  # 기본값 4.0에서 증가

# 더 구체적인 프롬프트 사용
# Bad: "change color"
# Good: "Change the background color to light sky blue (#87CEEB) while keeping the person intact"
```

#### 4. 이미지 품질 저하

**해결 방법:**

```python
# 추론 스텝 증가
inputs["num_inference_steps"] = 50  # 기본값 40에서 증가

# negative prompt 추가
inputs["negative_prompt"] = "blurry, low quality, pixelated, noisy, artifacts"
```

#### 5. Windows에서 발생하는 문제

**긴 경로 오류 (Long Path):**

```powershell
# LongPathsEnabled 설정 (관리자 권한으로 PowerShell 실행 후)
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

**DLL 로드 오류:**

```
OSError: [WinError 126] 지정된 모듈을 찾을 수 없습니다
```

```powershell
# Visual C++ Redistributable 설치 필요
# https://aka.ms/vs/17/release/vc_redist.x64.exe 다운로드 후 설치

# 또는 winget으로 설치
winget install Microsoft.VCRedist.2015+.x64
```

**torch.compile 오류:**

```
Triton not found, skipping compilation
```

```python
# Windows에서는 torch.compile 생략
# pipeline.transformer = torch.compile(...)  # 주석 처리

# 대신 다른 최적화 방법 사용
pipeline.enable_attention_slicing("auto")
```

#### 6. Flash Attention 2 관련 오류

```
RuntimeError: FlashAttention only supports NVIDIA GPUs or ROCm
ModuleNotFoundError: No module named 'flash_attn'
```

**해결 방법:**

```python
# Flash Attention 2는 Windows에서 지원되지 않습니다.
# use_flash_attention_2=True 옵션을 제거하세요.

# ❌ 잘못된 코드 (Windows)
# pipeline = QwenImageEditPlusPipeline.from_pretrained(
#     "Qwen/Qwen-Image-Edit-2511",
#     use_flash_attention_2=True  # Windows에서 오류 발생
# )

# ✅ 올바른 코드 (Windows)
pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    torch_dtype=torch.bfloat16
)
pipeline.enable_attention_slicing("auto")  # 대안
```

#### 7. bitsandbytes 오류 (Windows)

```
RuntimeError: CUDA Setup failed despite GPU being available
```

**해결 방법:**

```powershell
# bitsandbytes 재설치
pip uninstall bitsandbytes
pip install bitsandbytes>=0.43.0

# 그래도 안 되면 → 사전 양자화 모델 사용 (권장)
```

**권장 대안 - ovedrive NF4 모델 사용:**

Windows에서 양자화 문제가 계속된다면 [ovedrive/Qwen-Image-Edit-2511-4bit](https://huggingface.co/ovedrive/Qwen-Image-Edit-2511-4bit)를 사용하세요. 이 모델은 이미 양자화되어 있어 bitsandbytes 설정이 필요 없습니다.

```python
from diffusers import QwenImageEditPlusPipeline
import torch

# 사전 양자화된 모델 - bitsandbytes 불필요
pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "ovedrive/Qwen-Image-Edit-2511-4bit",
    torch_dtype=torch.bfloat16
)
pipeline.enable_model_cpu_offload()  # 16GB VRAM
```

| GPU VRAM | 권장 설정 |
|----------|-----------|
| 20GB+ | `pipeline.to("cuda")` |
| 16-20GB | `pipeline.enable_model_cpu_offload()` |

---

## 라이선스 및 인용

### 라이선스

Qwen-Image-Edit-2511은 **Apache 2.0** 라이선스 하에 제공됩니다.

- ✅ 상업적 사용 가능
- ✅ 수정 및 배포 가능
- ✅ 특허권 부여
- ⚠️ 라이선스 및 저작권 고지 필수
- ⚠️ 보증 없음

### 인용 (Citation)

연구나 프로젝트에서 이 모델을 사용할 경우, 다음과 같이 인용해 주세요:

```bibtex
@misc{wu2025qwenimagetechnicalreport,
      title={Qwen-Image Technical Report}, 
      author={Chenfei Wu and Jiahao Li and Jingren Zhou and Junyang Lin and 
              Kaiyuan Gao and Kun Yan and Sheng-ming Yin and Shuai Bai and 
              Xiao Xu and Yilei Chen and Yuxiang Chen and Zecheng Tang and 
              Zekai Zhang and Zhengyi Wang and An Yang and Bowen Yu and 
              Chen Cheng and Dayiheng Liu and Deqing Li and Hang Zhang and 
              Hao Meng and Hu Wei and Jingyuan Ni and Kai Chen and Kuan Cao and 
              Liang Peng and Lin Qu and Minggang Wu and Peng Wang and Shuting Yu and 
              Tingkun Wen and Wensen Feng and Xiaoxiao Xu and Yi Wang and 
              Yichang Zhang and Yongqiang Zhu and Yujia Wu and Yuxuan Cai and 
              Zenan Liu},
      year={2025},
      eprint={2508.02324},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.02324}, 
}
```

---

## 참고 자료

### 공식 링크

| 리소스 | URL |
|--------|-----|
| 🤗 Hugging Face | https://huggingface.co/Qwen/Qwen-Image-Edit-2511 |
| 📄 기술 보고서 (arXiv) | https://arxiv.org/abs/2508.02324 |
| 💬 Qwen Chat | https://chat.qwen.ai |
| 🖥️ 온라인 데모 | https://huggingface.co/spaces/Qwen/Qwen-Image-Edit-2511 |
| 🐙 GitHub | https://github.com/QwenLM |

### 커뮤니티 리소스

| 리소스 | 설명 | URL |
|--------|------|-----|
| **NF4 양자화 (ovedrive)** | 16GB VRAM용 NF4 양자화 모델 (권장) | [ovedrive/Qwen-Image-Edit-2511-4bit](https://huggingface.co/ovedrive/Qwen-Image-Edit-2511-4bit) |
| Fast 버전 | 빠른 추론을 위한 최적화 버전 | [linoyts/Qwen-Image-Edit-2511-Fast](https://huggingface.co/spaces/linoyts/Qwen-Image-Edit-2511-Fast) |
| Lightning 버전 | 4-step 추론 버전 | [akhaliq/Qwen-Image-Edit-2511-Lightning](https://huggingface.co/spaces/akhaliq/Qwen-Image-Edit-2511-Lightning) |
| AnyPose 버전 | 포즈 제어 기능 추가 | [linoyts/Qwen-Image-Edit-2511-AnyPose](https://huggingface.co/spaces/linoyts/Qwen-Image-Edit-2511-AnyPose) |
| RunComfy API | API 통합 | [runcomfy.com](https://www.runcomfy.com/models/qwen/qwen-image/qwen-image-edit-2511) |
| WaveSpeed API | 빠른 추론 API | [wavespeed.ai](https://wavespeed.ai/models/wavespeed-ai/qwen-image/edit-2511) |

### 관련 문서

- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Hugging Face Hub](https://huggingface.co/docs/hub)

---

## 버전 이력

| 버전 | 날짜 | 주요 변경 사항 |
|------|------|----------------|
| 2511 | 2025년 11월 | 캐릭터 일관성 향상, LoRA 통합, 기하학적 추론 강화 |
| 2509 | 2025년 9월 | 초기 릴리스 |

---

<div align="center">

**Made with ❤️ by Qwen Team**

[Hugging Face](https://huggingface.co/Qwen) • [GitHub](https://github.com/QwenLM) • [Discord](https://discord.gg/qwen)

</div>

