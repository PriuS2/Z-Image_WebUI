# 🎨 Z-Image WebUI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Z-Image Turbo 모델을 활용한 빠르고 고품질의 AI 이미지 생성 웹 인터페이스**

[🚀 빠른 시작](#-빠른-시작) • [✨ 주요 기능](#-주요-기능) • [📦 설치 가이드](#-설치-가이드) • [🎯 사용 방법](#-사용-방법) • [⚙️ 설정](#️-설정)

</div>

---

## 📖 소개

Z-Image WebUI는 [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) 모델을 기반으로 한 FastAPI 웹 애플리케이션입니다. 한국어 프롬프트 지원, GGUF 양자화를 통한 VRAM 절약, AI 프롬프트 향상 등 다양한 기능을 제공합니다.

### 왜 Z-Image WebUI인가?

- ⚡ **빠른 생성**: Turbo 모델로 8~9 스텝만에 고품질 이미지 생성
- 🇰🇷 **한국어 지원**: 한국어 프롬프트 자동 번역 및 역번역
- 💾 **VRAM 절약**: GGUF 양자화로 3.7GB~7.2GB VRAM만으로 실행 가능
- 🤖 **AI 프롬프트 향상**: LLM을 활용한 프롬프트 자동 개선
- 🖼️ **업스케일링**: Real-ESRGAN 기반 고화질 업스케일링

---

## ✨ 주요 기능

### 🎨 이미지 생성
- **다양한 해상도**: 512x512부터 1024x1024까지 프리셋 제공
- **시드 컨트롤**: 재현 가능한 결과를 위한 시드 관리
- **배치 생성**: 한 번에 여러 이미지 생성
- **미리보기**: 256x256 빠른 미리보기 모드

### 🌐 프롬프트 도구
- **자동 번역**: 한국어 → 영어 실시간 번역
- **역번역**: 영어 → 한국어 프롬프트 번역
- **AI 향상**: 간단한 프롬프트를 상세하게 확장
- **템플릿**: 인물, 풍경, 애니메이션 등 미리 정의된 템플릿

### 💾 모델 관리
- **BF16 기본 모델**: 최고 품질 (약 12GB VRAM)
- **GGUF 양자화**: Q3~Q8까지 다양한 옵션
  - Q4_K_M (4.98GB) - 품질과 VRAM의 균형, **추천**
  - Q8_0 (7.22GB) - 고품질
  - Q3_K_S (3.79GB) - 최저 VRAM
- **자동 언로드**: 비활성 시 자동으로 모델 언로드하여 VRAM 절약

### 🖼️ 갤러리 & 관리
- **갤러리**: 생성된 이미지 썸네일 뷰
- **메타데이터**: 각 이미지에 프롬프트, 시드 등 메타데이터 저장
- **히스토리**: 프롬프트 사용 기록 관리
- **즐겨찾기**: 자주 사용하는 프롬프트 저장

### 🔧 LLM Provider 지원
- OpenAI (GPT-4o, GPT-4o-mini 등)
- Groq (Llama, Mixtral)
- Together AI
- Ollama (로컬)
- LM Studio (로컬)
- OpenRouter
- 커스텀 OpenAI 호환 API

---

## 🚀 빠른 시작

### Windows 사용자

```batch
# 1. Setup.bat 실행 (가상환경 생성 + 패키지 설치)
Setup.bat

# 2. Run.bat 실행 (WebUI 시작)
Run.bat
```

### Linux/macOS 사용자

```bash
# 1. 가상환경 생성
python -m venv venv
source venv/bin/activate

# 2. 패키지 설치
pip install -r requirements.txt

# 3. 실행
python app.py
```

브라우저에서 **http://localhost:7860** 접속

---

## 📦 설치 가이드

### 시스템 요구 사항

| 구성 요소 | 최소 사양 | 권장 사양 |
|----------|----------|----------|
| Python | 3.8+ | 3.10+ |
| VRAM | 4GB (Q3 양자화) | 8GB+ (Q4~Q6) |
| RAM | 16GB | 32GB |
| 저장 공간 | 20GB | 50GB |

### 1. 저장소 클론

```bash
git clone https://github.com/your-username/Z-Image_WebUI.git
cd Z-Image_WebUI
```

### 2. 가상환경 설정

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

> ⚠️ **중요**: GGUF 양자화 모델 지원을 위해 diffusers GitHub 버전이 필요합니다.
> requirements.txt에 이미 포함되어 있습니다.

### 4. (선택) Real-ESRGAN 설치

업스케일링 기능을 사용하려면:

```bash
pip install -r requirements-upscale.txt
```

> Windows에서 빌드 오류 발생 시 [RUN_WEBUI.md](RUN_WEBUI.md)의 해결 방법 참조

---

## 🎯 사용 방법

### 1. 모델 로드

1. 웹 UI에서 **설정** 탭으로 이동
2. 양자화 옵션 선택 (VRAM에 따라)
3. **모델 로드** 클릭
4. 첫 실행 시 모델 다운로드 (약 5~12GB)

### 2. 이미지 생성

1. **프롬프트** 입력 (한국어 또는 영어)
2. 해상도, 스텝 수 설정
3. **생성** 버튼 클릭

### 3. 프롬프트 도구 활용

| 버튼 | 기능 |
|-----|------|
| 🌐 번역 | 한국어 → 영어 번역 |
| ⬅️ 역번역 | 영어 → 한국어 번역 |
| ✨ AI 향상 | 프롬프트 상세화 |
| 📋 템플릿 | 미리 정의된 템플릿 |

### 4. 미리보기 모드

- 빠른 테스트를 위해 256x256 저해상도로 생성
- 프롬프트 확인 후 본 생성 진행

---

## ⚙️ 설정

### LLM API 설정 (번역/향상 기능)

1. **설정** 탭에서 LLM Provider 선택
2. API 키 입력
3. (선택) 커스텀 Base URL, 모델 지정

**지원 Provider:**
- OpenAI: `api.openai.com`
- Groq: `api.groq.com`
- Together AI: `api.together.xyz`
- Ollama: `localhost:11434` (로컬)
- LM Studio: `localhost:1234` (로컬)

### 자동 언로드 설정

VRAM 절약을 위해 비활성 시 모델 자동 언로드:

- **활성화/비활성화**: 설정에서 토글
- **타임아웃**: 1~1440분 (기본 10분)

### 파일명 패턴

생성 이미지 파일명 커스터마이징:

| 패턴 | 예시 |
|-----|------|
| `{date}_{time}_{seed}` | 20241211_143052_123456.png |
| `{prompt_short}_{seed}` | beautiful_landscape_123456.png |
| `image_{counter}_{seed}` | image_001_123456.png |

---

## 📁 프로젝트 구조

```
Z-Image_WebUI/
├── app.py                 # FastAPI 메인 애플리케이션
├── inference.py           # 스탠드얼론 추론 스크립트
├── config/
│   ├── defaults.py        # 기본 설정값, 양자화 옵션
│   └── templates.py       # 프롬프트 템플릿
├── utils/
│   ├── llm_client.py      # LLM API 클라이언트
│   ├── translator.py      # 번역 유틸리티
│   ├── prompt_enhancer.py # AI 프롬프트 향상
│   ├── metadata.py        # 이미지 메타데이터 처리
│   ├── history.py         # 히스토리 관리
│   ├── favorites.py       # 즐겨찾기 관리
│   ├── settings.py        # 설정 관리
│   └── upscaler.py        # Real-ESRGAN 업스케일링
├── static/
│   ├── css/style.css      # 스타일시트
│   └── js/app.js          # 프론트엔드 JavaScript
├── templates/
│   └── index.html         # 메인 HTML 템플릿
├── data/                  # 사용자 데이터 (자동 생성)
│   ├── settings.yaml
│   ├── history.json
│   └── favorites.json
├── outputs/               # 생성된 이미지 저장
├── requirements.txt       # 메인 의존성
├── requirements-upscale.txt # 업스케일링 의존성
├── Setup.bat              # Windows 설치 스크립트
└── Run.bat                # Windows 실행 스크립트
```

---

## 🔧 문제 해결

### 모델 로드 실패

```bash
# diffusers 최신 버전 재설치
pip install --force-reinstall git+https://github.com/huggingface/diffusers
```

### VRAM 부족

1. 더 낮은 양자화 옵션 선택 (Q4_K_S, Q3_K_M 등)
2. CPU 오프로딩 활성화
3. 해상도 낮추기

### 번역/향상 기능 미작동

1. 설정에서 LLM API 키 확인
2. API 키 잔액/유효성 확인
3. 네트워크 연결 확인

### Windows에서 Real-ESRGAN 설치 오류

```bash
# Visual Studio Build Tools 설치 후 시도
# 또는 --no-deps 옵션으로 설치
pip install realesrgan --no-deps
pip install facexlib gfpgan
```

---

## 🛠️ 개발

### 스탠드얼론 추론

WebUI 없이 커맨드라인에서 이미지 생성:

```bash
python inference.py
```

`inference.py`에서 프롬프트와 설정 수정 후 실행

### API 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|-----------|-------|------|
| `/api/status` | GET | 시스템 상태 |
| `/api/model/load` | POST | 모델 로드 |
| `/api/model/unload` | POST | 모델 언로드 |
| `/api/generate` | POST | 이미지 생성 |
| `/api/translate` | POST | 프롬프트 번역 |
| `/api/enhance` | POST | 프롬프트 향상 |
| `/api/gallery` | GET | 갤러리 목록 |
| `/api/history` | GET | 히스토리 목록 |
| `/api/favorites` | GET/POST/DELETE | 즐겨찾기 관리 |
| `/ws` | WebSocket | 실시간 상태 업데이트 |

---

## 📝 라이선스

이 프로젝트는 MIT 라이선스로 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

## 🙏 크레딧

- [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) - 이미지 생성 모델
- [jayn7/Z-Image-Turbo-GGUF](https://huggingface.co/jayn7/Z-Image-Turbo-GGUF) - GGUF 양자화 모델
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) - 파이프라인 프레임워크
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - 업스케일링

---

<div align="center">

**Made with ❤️ for AI Art Creators**

</div>

