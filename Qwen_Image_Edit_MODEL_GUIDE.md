# ovedrive/Qwen-Image-Edit-2511-4bit 모델 사용 가이드

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-ovedrive%2FQwen--Image--Edit--2511--4bit-blue)](https://huggingface.co/ovedrive/Qwen-Image-Edit-2511-4bit)

> **ovedrive/Qwen-Image-Edit-2511-4bit**는 Qwen-Image-Edit-2511의 **NF4 양자화 버전**으로, 원본 모델 대비 절반 이하의 VRAM으로 이미지 편집이 가능합니다.

> 📌 이 문서는 **Windows 환경**을 기준으로 작성되었으며, 이 프로젝트에서의 사용법을 중점적으로 다룹니다.

---

## 📋 목차

- [모델 개요](#모델-개요)
- [시스템 요구사항](#시스템-요구사항)
- [모델 다운로드](#모델-다운로드)
- [모델 로드](#모델-로드)
- [모델 언로드](#모델-언로드)
- [사용 방법](#사용-방법)
- [최적화 방법](#최적화-방법)
- [문제 해결](#문제-해결)

---

## 모델 개요

### 기본 정보

| 항목 | 내용 |
|------|------|
| **모델 ID** | `ovedrive/Qwen-Image-Edit-2511-4bit` |
| **원본 모델** | `Qwen/Qwen-Image-Edit-2511` |
| **양자화 방식** | NF4 (4-bit Normal Float) |
| **파이프라인** | `QwenImageEditPlusPipeline` |
| **권장 dtype** | `bfloat16` |

### 왜 4bit 양자화 모델인가?

| 비교 항목 | 원본 (Qwen-Image-Edit-2511) | 4bit (ovedrive) |
|-----------|----------------------------|-----------------|
| 모델 크기 | ~40GB | ~8.5GB |
| VRAM 요구량 | 24GB+ | 16GB (CPU Offload 시) |
| VRAM 요구량 (Full GPU) | 24GB+ | ~20GB |
| 품질 | 기준 | 거의 동일 (미세한 차이) |
| 추론 속도 | 기준 | 약간 빠름 |
| Windows 호환성 | bitsandbytes 문제 가능 | ✅ 문제 없음 |

### 주요 장점

- ✅ **낮은 VRAM 사용량**: 16GB GPU에서도 실행 가능
- ✅ **Windows 호환성**: bitsandbytes 설정 불필요 (이미 양자화됨)
- ✅ **diffusers 호환**: 표준 파이프라인으로 바로 사용
- ✅ **빠른 로딩**: 더 작은 모델 크기로 빠른 로드
- ✅ **10 스텝에서도 작동**: 낮은 추론 스텝에서도 품질 유지

---

## 시스템 요구사항

### 최소 사양

| 구성 요소 | 최소 사양 | 권장 사양 |
|-----------|-----------|-----------|
| **GPU** | NVIDIA RTX 3060 (12GB) | NVIDIA RTX 3090/4070+ |
| **VRAM** | 12GB (CPU Offload 필수) | 16GB+ |
| **RAM** | 24GB | 32GB+ |
| **저장 공간** | 15GB (모델) | 25GB+ (캐시 포함) |
| **Python** | 3.10+ | 3.11+ |
| **CUDA** | 11.8+ | 12.1+ |

### VRAM별 권장 설정

| GPU VRAM | 권장 설정 | 설명 |
|----------|-----------|------|
| **24GB+** | `pipeline.to("cuda")` | 전체 GPU 로드, 최고 속도 |
| **16-24GB** | `enable_model_cpu_offload()` | CPU 오프로딩, 빠른 속도 |
| **12-16GB** | CPU Offload + Attention Slicing | 메모리 절약, 중간 속도 |

---

## 모델 다운로드

### 방법 1: API를 통한 다운로드 (권장)

이 프로젝트의 백엔드 서버를 통해 다운로드할 수 있습니다.

#### REST API 사용

```bash
# 다운로드 시작
curl -X POST "http://localhost:8000/api/model/download" \
  -H "X-API-Key: qwen-image-edit-default-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "ovedrive/Qwen-Image-Edit-2511-4bit",
    "force_download": false
  }'
```

```bash
# 다운로드 상태 확인
curl "http://localhost:8000/api/model/download/status" \
  -H "X-API-Key: qwen-image-edit-default-key"
```

#### 응답 예시

```json
{
  "success": true,
  "message": "Download status: downloading",
  "data": {
    "status": "downloading",
    "model_name": "ovedrive/Qwen-Image-Edit-2511-4bit",
    "progress_percent": 45.5,
    "downloaded_size_mb": 3890.5,
    "total_size_mb": 8550.0,
    "current_file": "model-00003-of-00005.safetensors",
    "files_completed": 2,
    "files_total": 5,
    "error_message": null
  }
}
```

#### Python 클라이언트 예시

```python
import requests
import time

API_URL = "http://localhost:8000"
API_KEY = "qwen-image-edit-default-key"
headers = {"X-API-Key": API_KEY}

# 다운로드 시작
response = requests.post(
    f"{API_URL}/api/model/download",
    headers=headers,
    json={"model_name": "ovedrive/Qwen-Image-Edit-2511-4bit"}
)
print(response.json())

# 진행 상황 확인
while True:
    response = requests.get(f"{API_URL}/api/model/download/status", headers=headers)
    data = response.json()["data"]
    
    print(f"📥 진행률: {data['progress_percent']:.1f}%")
    
    if data["status"] == "completed":
        print("✅ 다운로드 완료!")
        break
    elif data["status"] == "failed":
        print(f"❌ 다운로드 실패: {data['error_message']}")
        break
    
    time.sleep(2)
```

### 방법 2: Hugging Face CLI 다운로드

```powershell
# huggingface-cli 설치
pip install huggingface_hub

# 모델 다운로드
huggingface-cli download ovedrive/Qwen-Image-Edit-2511-4bit
```

### 방법 3: Python 직접 다운로드

```python
from huggingface_hub import snapshot_download

# 모델 전체 다운로드
cache_dir = snapshot_download(
    repo_id="ovedrive/Qwen-Image-Edit-2511-4bit",
    resume_download=True  # 중단 시 재개 가능
)

print(f"✅ 모델 캐시 위치: {cache_dir}")
```

### 다운로드 취소

```bash
curl -X POST "http://localhost:8000/api/model/download/cancel" \
  -H "X-API-Key: qwen-image-edit-default-key"
```

### 다운로드 여부 확인

```bash
# 특정 모델이 다운로드되어 있는지 확인
curl "http://localhost:8000/api/model/download/check/ovedrive%2FQwen-Image-Edit-2511-4bit" \
  -H "X-API-Key: qwen-image-edit-default-key"
```

```json
{
  "success": true,
  "model_name": "ovedrive/Qwen-Image-Edit-2511-4bit",
  "is_downloaded": true
}
```

---

## 모델 로드

### 방법 1: API를 통한 로드 (권장)

#### 기본 로드 (저장된 설정 사용)

```bash
curl -X POST "http://localhost:8000/api/model/load" \
  -H "X-API-Key: qwen-image-edit-default-key"
```

#### 커스텀 설정으로 로드

```bash
curl -X POST "http://localhost:8000/api/model/load" \
  -H "X-API-Key: qwen-image-edit-default-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "ovedrive/Qwen-Image-Edit-2511-4bit",
    "optimization": {
      "enable_model_cpu_offload": true,
      "enable_attention_slicing": true,
      "enable_vae_slicing": true,
      "enable_vae_tiling": false,
      "enable_xformers": false
    },
    "force_reload": false
  }'
```

#### 로드 옵션 설명

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `model_name` | string | 설정 값 | 로드할 모델 ID |
| `optimization` | object | 설정 값 | 최적화 옵션 |
| `force_reload` | boolean | `false` | 이미 로드되어 있어도 재로드 |

#### 응답 예시

```json
{
  "success": true,
  "message": "Model loaded successfully",
  "data": {
    "is_loaded": true,
    "model_name": "ovedrive/Qwen-Image-Edit-2511-4bit",
    "device": "cuda:0",
    "dtype": "bfloat16",
    "vram_used_gb": 8.5,
    "vram_total_gb": 24.0,
    "optimization": {
      "enable_model_cpu_offload": true,
      "enable_attention_slicing": true,
      "enable_vae_slicing": true,
      "enable_vae_tiling": false,
      "enable_xformers": false
    }
  }
}
```

### 방법 2: Python 직접 로드

```python
import torch
from diffusers import QwenImageEditPlusPipeline

# ═══════════════════════════════════════════════════════════════
# 기본 로드
# ═══════════════════════════════════════════════════════════════
pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "ovedrive/Qwen-Image-Edit-2511-4bit",
    torch_dtype=torch.bfloat16,
)

# ═══════════════════════════════════════════════════════════════
# VRAM 설정 (택 1)
# ═══════════════════════════════════════════════════════════════

# 옵션 A: 전체 GPU 로드 (24GB+ VRAM)
# pipeline.to("cuda")

# 옵션 B: CPU 오프로딩 (16GB+ VRAM, 권장)
pipeline.enable_model_cpu_offload()

# ═══════════════════════════════════════════════════════════════
# 추가 최적화 (선택)
# ═══════════════════════════════════════════════════════════════
pipeline.enable_attention_slicing("auto")
pipeline.enable_vae_slicing()

# 프로그레스 바 설정
pipeline.set_progress_bar_config(disable=None)

print("✅ 모델 로드 완료!")
```

### 로드 상태 확인

```bash
curl "http://localhost:8000/api/model/status" \
  -H "X-API-Key: qwen-image-edit-default-key"
```

### 자동 로드 기능

이 프로젝트는 **자동 로드** 기능을 지원합니다. 모델이 로드되지 않은 상태에서 이미지 편집 요청이 오면 자동으로 모델을 로드합니다.

```bash
# 자동 로드 설정 확인
curl "http://localhost:8000/api/settings/auto-load" \
  -H "X-API-Key: qwen-image-edit-default-key"
```

```bash
# 자동 로드 활성화
curl -X PUT "http://localhost:8000/api/settings/auto-load" \
  -H "X-API-Key: qwen-image-edit-default-key" \
  -H "Content-Type: application/json" \
  -d '{"enabled": true}'
```

---

## 모델 언로드

### 방법 1: API를 통한 언로드

```bash
curl -X POST "http://localhost:8000/api/model/unload" \
  -H "X-API-Key: qwen-image-edit-default-key"
```

#### 응답 예시

```json
{
  "success": true,
  "message": "Model unloaded successfully",
  "vram_freed_gb": 8.5
}
```

### 방법 2: Python 직접 언로드

```python
import gc
import torch

# 파이프라인 참조 제거
del pipeline

# Python 가비지 컬렉션
gc.collect()

# CUDA 캐시 정리
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

print("✅ 모델 언로드 완료!")
```

### 자동 언로드 기능

이 프로젝트는 **자동 언로드** 기능을 지원합니다. 일정 시간 동안 모델이 사용되지 않으면 자동으로 언로드하여 VRAM을 해제합니다.

```bash
# 자동 언로드 설정 확인
curl "http://localhost:8000/api/settings/auto-unload" \
  -H "X-API-Key: qwen-image-edit-default-key"
```

```bash
# 자동 언로드 설정 변경 (30분 → 60분)
curl -X PUT "http://localhost:8000/api/settings/auto-unload" \
  -H "X-API-Key: qwen-image-edit-default-key" \
  -H "Content-Type: application/json" \
  -d '{
    "enabled": true,
    "timeout_minutes": 60
  }'
```

---

## 사용 방법

### 방법 1: API를 통한 이미지 편집 (권장)

#### 파일 업로드 방식

```bash
curl -X POST "http://localhost:8000/api/edit/upload/single" \
  -H "X-API-Key: qwen-image-edit-default-key" \
  -F "image=@photo.jpg" \
  -F "prompt=Change the background to a sunset sky" \
  -F "num_inference_steps=20" \
  -F "true_cfg_scale=4.0"
```

#### JSON 방식 (Base64)

```bash
curl -X POST "http://localhost:8000/api/edit/single" \
  -H "X-API-Key: qwen-image-edit-default-key" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/png;base64,iVBORw0KGgoAAAANS...",
    "params": {
      "prompt": "Make the sky purple and add stars",
      "negative_prompt": " ",
      "num_inference_steps": 20,
      "true_cfg_scale": 4.0,
      "guidance_scale": 1.0,
      "seed": -1,
      "num_images_per_prompt": 1
    },
    "response_format": "url",
    "save_to_gallery": true
  }'
```

#### Python 클라이언트 완전 예시

```python
import requests
import time
import base64
from pathlib import Path

API_URL = "http://localhost:8000"
API_KEY = "qwen-image-edit-default-key"
headers = {"X-API-Key": API_KEY}

# ═══════════════════════════════════════════════════════════════
# 1. 모델 상태 확인 및 로드
# ═══════════════════════════════════════════════════════════════
response = requests.get(f"{API_URL}/api/model/status", headers=headers)
status = response.json()["data"]

if not status["is_loaded"]:
    print("🔄 모델 로딩 중...")
    response = requests.post(f"{API_URL}/api/model/load", headers=headers)
    print(f"✅ {response.json()['message']}")
else:
    print(f"✅ 모델 이미 로드됨: {status['model_name']}")

# ═══════════════════════════════════════════════════════════════
# 2. 이미지 편집 요청
# ═══════════════════════════════════════════════════════════════
with open("input_photo.jpg", "rb") as f:
    files = {"image": ("photo.jpg", f, "image/jpeg")}
    data = {
        "prompt": "Transform this photo into Studio Ghibli animation style",
        "num_inference_steps": 20,
        "true_cfg_scale": 4.0,
    }
    response = requests.post(
        f"{API_URL}/api/edit/upload/single",
        headers=headers,
        files=files,
        data=data
    )

result = response.json()
job_id = result["job_id"]
print(f"📤 작업 제출됨: {job_id}")

# ═══════════════════════════════════════════════════════════════
# 3. 작업 완료 대기
# ═══════════════════════════════════════════════════════════════
while True:
    response = requests.get(f"{API_URL}/api/edit/job/{job_id}", headers=headers)
    job_status = response.json()
    
    print(f"⏳ 진행률: {job_status['progress']}%")
    
    if job_status["status"] == "completed":
        result = job_status["result"]
        image_url = f"{API_URL}{result['image']}"
        print(f"✅ 완료! 이미지: {image_url}")
        
        # 이미지 다운로드
        img_response = requests.get(image_url)
        with open("output.png", "wb") as f:
            f.write(img_response.content)
        print("💾 output.png 저장 완료!")
        break
        
    elif job_status["status"] == "failed":
        print(f"❌ 실패: {job_status['error']}")
        break
    
    time.sleep(1)
```

#### WebSocket을 통한 실시간 진행률

```javascript
// JavaScript 예시
const ws = new WebSocket(`ws://localhost:8000/ws/progress/${jobId}`);

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`진행률: ${data.progress}%`);
  
  if (data.status === 'completed') {
    console.log('완료!', data.result);
    ws.close();
  }
};

// Keep-alive
setInterval(() => {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send('ping');
  }
}, 25000);
```

### 방법 2: Python 직접 사용

#### 기본 사용법

```python
import os
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

# ═══════════════════════════════════════════════════════════════
# 1. 파이프라인 로드
# ═══════════════════════════════════════════════════════════════
pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "ovedrive/Qwen-Image-Edit-2511-4bit",
    torch_dtype=torch.bfloat16,
)
pipeline.enable_model_cpu_offload()  # 16GB VRAM
pipeline.set_progress_bar_config(disable=None)
print("✅ 파이프라인 로드 완료")

# ═══════════════════════════════════════════════════════════════
# 2. 이미지 로드 및 편집
# ═══════════════════════════════════════════════════════════════
image = Image.open("input.png").convert("RGB")

inputs = {
    "image": [image],
    "prompt": "Change the background to a sunset sky",
    "generator": torch.manual_seed(42),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 20,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}

# ═══════════════════════════════════════════════════════════════
# 3. 추론 및 저장
# ═══════════════════════════════════════════════════════════════
with torch.inference_mode():
    output = pipeline(**inputs)

output_image = output.images[0]
output_image.save("output.png")
print(f"✅ 저장 완료: {os.path.abspath('output.png')}")
```

#### 다중 이미지 합성 예시

```python
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "ovedrive/Qwen-Image-Edit-2511-4bit",
    torch_dtype=torch.bfloat16,
)
pipeline.enable_model_cpu_offload()

# 두 개의 인물 이미지
person1 = Image.open("person1.png").convert("RGB")
person2 = Image.open("person2.png").convert("RGB")

inputs = {
    "image": [person1, person2],  # 최대 3개까지 가능
    "prompt": "The two people are sitting together at a coffee shop table, having a friendly conversation",
    "generator": torch.manual_seed(123),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 20,
    "guidance_scale": 1.0,
}

with torch.inference_mode():
    output = pipeline(**inputs)

output.images[0].save("combined_scene.png")
print("✅ 합성 이미지 저장 완료!")
```

#### 스타일 변환 예시

```python
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "ovedrive/Qwen-Image-Edit-2511-4bit",
    torch_dtype=torch.bfloat16,
)
pipeline.enable_model_cpu_offload()

image = Image.open("landscape.png").convert("RGB")

# 다양한 스타일 변환 프롬프트
styles = [
    ("ghibli", "Transform this into Studio Ghibli animation style"),
    ("oil_painting", "Convert this photo to an oil painting style"),
    ("cyberpunk", "Transform this into a cyberpunk neon city style"),
    ("watercolor", "Convert this to a watercolor painting"),
]

for style_name, prompt in styles:
    inputs = {
        "image": [image],
        "prompt": prompt,
        "generator": torch.manual_seed(42),
        "true_cfg_scale": 5.0,
        "negative_prompt": "blurry, distorted, low quality",
        "num_inference_steps": 25,
        "guidance_scale": 1.0,
    }
    
    with torch.inference_mode():
        output = pipeline(**inputs)
    
    output.images[0].save(f"output_{style_name}.png")
    print(f"✅ {style_name} 스타일 저장 완료")
```

### 파라미터 가이드

| 파라미터 | 타입 | 기본값 | 범위 | 설명 |
|----------|------|--------|------|------|
| `image` | List[PIL.Image] | 필수 | 1-3개 | 입력 이미지 리스트 |
| `prompt` | str | 필수 | - | 편집 지시 프롬프트 |
| `negative_prompt` | str | `" "` | - | 제외할 요소 |
| `num_inference_steps` | int | `20` | 1-100 | 추론 스텝 수 |
| `true_cfg_scale` | float | `4.0` | 1.0-20.0 | 프롬프트 충실도 |
| `guidance_scale` | float | `1.0` | 0.0-20.0 | 가이던스 스케일 |
| `seed` / `generator` | int / Generator | -1 (랜덤) | - | 재현성을 위한 시드 |
| `num_images_per_prompt` | int | `1` | 1-4 | 생성할 이미지 수 |

### 스텝 수 가이드

| 스텝 수 | 품질 | 속도 | 용도 |
|---------|------|------|------|
| 10-15 | 기본 | 매우 빠름 | 빠른 테스트/미리보기 |
| 20 | 좋음 | 빠름 | 일반 사용 (권장) |
| 25-30 | 높음 | 보통 | 고품질 결과물 |
| 40-50 | 매우 높음 | 느림 | 최종 결과물 |

---

## 최적화 방법

### 최적화 옵션 개요

| 옵션 | VRAM 절감 | 속도 영향 | Windows 지원 | 설명 |
|------|-----------|-----------|--------------|------|
| `enable_model_cpu_offload` | 높음 (-8GB) | -30% | ✅ | 사용하지 않는 레이어를 CPU로 이동 |
| `enable_attention_slicing` | 중간 (-4GB) | -10% | ✅ | Attention 연산 분할 |
| `enable_vae_slicing` | 낮음 (-2GB) | -5% | ✅ | VAE 연산 분할 |
| `enable_vae_tiling` | 중간 (-3GB) | -15% | ✅ | 대용량 이미지용 타일링 |
| `enable_xformers` | 중간 (-2GB) | +20% | ⚠️ 설치 주의 | 메모리 효율적 Attention |

### API를 통한 최적화 설정

#### 현재 최적화 설정 조회

```bash
curl "http://localhost:8000/api/model/optimization" \
  -H "X-API-Key: qwen-image-edit-default-key"
```

#### 최적화 설정 변경

```bash
curl -X PUT "http://localhost:8000/api/model/optimization" \
  -H "X-API-Key: qwen-image-edit-default-key" \
  -H "Content-Type: application/json" \
  -d '{
    "optimization": {
      "enable_model_cpu_offload": true,
      "enable_attention_slicing": true,
      "enable_vae_slicing": true,
      "enable_vae_tiling": false,
      "enable_xformers": false
    },
    "apply_immediately": true
  }'
```

### VRAM별 권장 설정

#### 24GB+ VRAM (RTX 4090, A100)

```python
# 전체 GPU 로드 - 최고 속도
pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "ovedrive/Qwen-Image-Edit-2511-4bit",
    torch_dtype=torch.bfloat16,
)
pipeline.to("cuda")
```

```json
{
  "enable_model_cpu_offload": false,
  "enable_attention_slicing": false,
  "enable_vae_slicing": false,
  "enable_vae_tiling": false,
  "enable_xformers": false
}
```

#### 16-20GB VRAM (RTX 3090, 4070 Ti Super)

```python
# CPU 오프로딩 - 균형 잡힌 설정
pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "ovedrive/Qwen-Image-Edit-2511-4bit",
    torch_dtype=torch.bfloat16,
)
pipeline.enable_model_cpu_offload()
pipeline.enable_attention_slicing("auto")
```

```json
{
  "enable_model_cpu_offload": true,
  "enable_attention_slicing": true,
  "enable_vae_slicing": true,
  "enable_vae_tiling": false,
  "enable_xformers": false
}
```

#### 12-16GB VRAM (RTX 3060 12GB, 4060 Ti 16GB)

```python
# 최대 메모리 절약
pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "ovedrive/Qwen-Image-Edit-2511-4bit",
    torch_dtype=torch.bfloat16,
)
pipeline.enable_model_cpu_offload()
pipeline.enable_attention_slicing("max")
pipeline.enable_vae_slicing()
pipeline.enable_vae_tiling()
```

```json
{
  "enable_model_cpu_offload": true,
  "enable_attention_slicing": true,
  "enable_vae_slicing": true,
  "enable_vae_tiling": true,
  "enable_xformers": false
}
```

### Windows에서 사용 불가능한 최적화

❌ **Flash Attention 2**: Linux 전용  
⚠️ **torch.compile**: Windows에서 제한적 (Triton 미지원)  
⚠️ **xFormers**: 설치 시 버전 호환성 주의

### 종합 최적화 예시 (Windows)

```python
import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image

def create_optimized_pipeline(vram_gb: int = 16):
    """
    VRAM에 따른 최적화된 파이프라인 생성
    
    Args:
        vram_gb: 사용 가능한 VRAM (GB)
    """
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "ovedrive/Qwen-Image-Edit-2511-4bit",
        torch_dtype=torch.bfloat16,
    )
    
    if vram_gb >= 24:
        # 고성능: 전체 GPU
        pipeline.to("cuda")
        print("🚀 전체 GPU 모드")
    elif vram_gb >= 16:
        # 균형: CPU 오프로딩
        pipeline.enable_model_cpu_offload()
        pipeline.enable_attention_slicing("auto")
        pipeline.enable_vae_slicing()
        print("⚡ CPU 오프로딩 모드")
    else:
        # 저메모리: 최대 절약
        pipeline.enable_model_cpu_offload()
        pipeline.enable_attention_slicing("max")
        pipeline.enable_vae_slicing()
        pipeline.enable_vae_tiling()
        print("💾 저메모리 모드")
    
    # xFormers 시도 (설치된 경우)
    try:
        pipeline.enable_xformers_memory_efficient_attention()
        print("✅ xFormers 활성화")
    except Exception:
        print("⚠️ xFormers 미설치 → Attention Slicing 사용")
    
    pipeline.set_progress_bar_config(disable=None)
    return pipeline

# 사용 예시
pipeline = create_optimized_pipeline(vram_gb=16)

image = Image.open("input.png").convert("RGB")
prompt = "Transform this photo into anime style"

with torch.inference_mode():
    output = pipeline(
        image=[image],
        prompt=prompt,
        generator=torch.manual_seed(42),
        true_cfg_scale=4.0,
        negative_prompt=" ",
        num_inference_steps=20,
    )

output.images[0].save("output.png")
print("✅ 완료!")
```

---

## 문제 해결

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**해결 방법:**

```python
# 1. CPU 오프로딩 활성화
pipeline.enable_model_cpu_offload()

# 2. 메모리 정리
import gc
import torch
gc.collect()
torch.cuda.empty_cache()

# 3. 이미지 크기 축소
from PIL import Image
image = Image.open("large_image.png")
image = image.resize((512, 512))

# 4. 추론 스텝 감소
inputs["num_inference_steps"] = 15
```

### 모델 로드 실패

```
OSError: Can't load the model
```

**해결 방법:**

```powershell
# diffusers 최신 버전 설치
pip install --upgrade git+https://github.com/huggingface/diffusers

# 캐시 정리
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface\hub\models--ovedrive--Qwen-Image-Edit-2511-4bit"

# 재다운로드
huggingface-cli download ovedrive/Qwen-Image-Edit-2511-4bit
```

### 프롬프트가 반영되지 않음

**해결 방법:**

```python
# true_cfg_scale 증가
inputs["true_cfg_scale"] = 5.0  # 기본값 4.0에서 증가

# 더 구체적인 프롬프트 사용
# ❌ 나쁜 예: "change color"
# ✅ 좋은 예: "Change the background to a bright blue sky with white clouds"
```

### Windows 긴 경로 오류

```powershell
# 관리자 권한으로 실행
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

### DLL 로드 오류

```powershell
# Visual C++ Redistributable 설치
winget install Microsoft.VCRedist.2015+.x64
```

---

## 관련 링크

| 리소스 | URL |
|--------|-----|
| 🤗 모델 페이지 | https://huggingface.co/ovedrive/Qwen-Image-Edit-2511-4bit |
| 🤗 원본 모델 | https://huggingface.co/Qwen/Qwen-Image-Edit-2511 |
| 📚 Diffusers 문서 | https://huggingface.co/docs/diffusers |
| 📄 이 프로젝트 API 문서 | [API_DOCS.md](backend/API_DOCS.md) |
| 📄 원본 모델 가이드 | [QWEN-IMAGE-EDIT-2511.md](QWEN-IMAGE-EDIT-2511.md) |

---

<div align="center">

**Made for Qwen-Image-Edit-WebUI Project**

</div>
