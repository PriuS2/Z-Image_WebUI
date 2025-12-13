# Z-Image WebUI 실행 가이드

## 📦 설치

### 1. 필수 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. (선택) Real-ESRGAN 설치 (업스케일링용)

**Linux/macOS:**
```bash
pip install -r requirements-upscale.txt
```

**Windows:** (빌드 오류 발생 시)
```bash
# 방법 1: Visual Studio Build Tools 설치 후 시도
pip install -r requirements-upscale.txt

# 방법 2: 빌드 없이 설치 (권장)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install realesrgan --no-deps
pip install facexlib gfpgan

# 방법 3: 업스케일링 없이 사용 (Lanczos 리사이즈로 대체됨)
# → 별도 설치 없이 앱 실행 가능
```

> ⚠️ Real-ESRGAN 미설치 시에도 앱은 정상 작동합니다.  
> 업스케일링 시 Lanczos 리사이즈로 자동 대체됩니다.

## 🚀 실행

```bash
python app.py
```

브라우저에서 자동으로 열리며, 기본 주소는: **http://localhost:7860**

## 🎯 기능 안내

### 이미지 생성 탭
- **프롬프트 입력**: 한국어 또는 영어로 입력 가능
- **자동 번역**: 한국어 입력 시 자동으로 영어로 번역 (OpenAI API 필요)
- **프롬프트 도구**:
  - 🌐 번역: 프롬프트를 영어로 번역
  - ✨ AI 향상: 프롬프트를 더 상세하게 개선
  - 템플릿: 미리 정의된 프롬프트 템플릿 사용
  - 가중치: 특정 키워드에 가중치 적용 `(keyword:1.5)`
- **미리보기**: 256x256 작은 이미지로 빠르게 테스트
- **즐겨찾기**: 자주 쓰는 프롬프트 저장/불러오기

### 갤러리 탭
- 생성된 이미지 목록 확인
- 이미지 메타데이터 (프롬프트, 시드 등) 확인
- Real-ESRGAN 업스케일링 (2x/4x)

### 히스토리 탭
- 사용한 프롬프트 기록 확인
- 이전 프롬프트 재사용

### 설정 탭
- **모델 설정**: 양자화 선택, CPU 오프로딩
- **API 설정**: OpenAI API 키 (번역/향상 기능용)
- **출력 설정**: 저장 폴더, 파일명 패턴, 테마
- **설정 내보내기/가져오기**: JSON 형식

## ⚙️ 첫 실행 시

1. **설정 탭**으로 이동
2. **모델 로드** 클릭 (첫 실행 시 모델 다운로드에 시간 소요)
3. (선택) **OpenAI API 키** 입력 → 저장

## 💡 팁

- **미리보기 기능**: 프롬프트 테스트 시 빠른 확인용
- **시드 고정**: 같은 결과 재현 시 시드 값 기록
- **VRAM 부족 시**: CPU 오프로딩 옵션 활성화
- **양자화**: VRAM이 부족하면 Q4 또는 Q5 양자화 선택

## 🔧 문제 해결

### 모델 로드 실패
- diffusers 최신 버전 필요: `pip install git+https://github.com/huggingface/diffusers`
- 인터넷 연결 확인

### 번역/향상 기능 미작동
- 설정 탭에서 OpenAI API 키 확인
- API 키 잔액/유효성 확인

### 업스케일링 미작동
- Real-ESRGAN 설치 확인
- 미설치 시 Lanczos 리사이즈로 대체됨

## 📁 파일 구조

```
Zimage/
├── app.py              # 메인 웹앱
├── config/
│   ├── defaults.py     # 기본 설정값
│   └── templates.py    # 프롬프트 템플릿
├── utils/
│   ├── translator.py   # 번역 기능
│   ├── prompt_enhancer.py  # AI 향상
│   ├── metadata.py     # 메타데이터 처리
│   ├── history.py      # 히스토리 관리
│   ├── favorites.py    # 즐겨찾기 관리
│   ├── settings.py     # 설정 관리
│   └── upscaler.py     # 업스케일링
├── data/               # 사용자 데이터 (자동 생성)
├── outputs/            # 생성 이미지 저장
└── requirements.txt    # 의존성 패키지
```
