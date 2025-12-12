"""OpenAI 호환 API 클라이언트 유틸리티"""

import base64
from io import BytesIO
from typing import Optional, Dict, Any, List, Union
from openai import OpenAI
from PIL import Image

from config.defaults import LLM_PROVIDER, LLM_API_KEY, LLM_MODEL, LLM_BASE_URL
from utils.settings import settings, LLM_PROVIDERS


class LLMClient:
    """다양한 OpenAI 호환 provider를 지원하는 LLM 클라이언트"""
    
    def __init__(self):
        self._client: Optional[OpenAI] = None
        self._current_config: Dict[str, Any] = {}
    
    def _get_config(self) -> Dict[str, Any]:
        """현재 설정에서 LLM 구성 가져오기"""
        # settings.yaml에서 provider 가져오기
        provider_id = settings.get("llm_provider", "env")
        
        # "env" provider: 환경변수(.env) 설정 사용
        if provider_id == "env":
            # 환경변수에서 실제 provider 결정
            actual_provider = LLM_PROVIDER or "openai"
            provider_info = LLM_PROVIDERS.get(actual_provider, LLM_PROVIDERS["openai"])
            
            api_key = LLM_API_KEY
            
            if actual_provider == "custom":
                base_url = LLM_BASE_URL
            else:
                base_url = LLM_BASE_URL or provider_info["base_url"]
            
            model = LLM_MODEL or provider_info["default_model"]
            
            return {
                "provider": "env",
                "actual_provider": actual_provider,
                "api_key": api_key,
                "base_url": base_url,
                "model": model,
            }
        
        # 다른 provider: 웹 UI 설정 사용
        provider_info = LLM_PROVIDERS.get(provider_id, LLM_PROVIDERS["openai"])
        
        # API 키: settings에서 가져오기
        api_key = settings.get("llm_api_key") or settings.get("openai_api_key", "")
        
        # Base URL: 커스텀이면 설정값, 아니면 provider 기본값
        if provider_id == "custom":
            base_url = settings.get("llm_base_url", "")
        elif provider_id in ["ollama", "lmstudio"]:
            base_url = settings.get("llm_base_url", "") or provider_info["base_url"]
        else:
            base_url = provider_info["base_url"]
        
        # 모델: settings 우선, 없으면 provider 기본값
        model = settings.get("llm_model") or provider_info["default_model"]
        
        return {
            "provider": provider_id,
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
        }
    
    def _should_recreate_client(self, config: Dict[str, Any]) -> bool:
        """클라이언트를 재생성해야 하는지 확인"""
        if self._client is None:
            return True
        
        # api_key 또는 base_url이 변경되면 재생성
        return (
            config["api_key"] != self._current_config.get("api_key") or
            config["base_url"] != self._current_config.get("base_url")
        )
    
    @property
    def client(self) -> Optional[OpenAI]:
        """OpenAI 호환 클라이언트 (지연 초기화)"""
        config = self._get_config()
        
        if not config["api_key"] and config["provider"] not in ["ollama", "lmstudio"]:
            return None
        
        if self._should_recreate_client(config):
            client_kwargs = {}
            
            if config["api_key"]:
                client_kwargs["api_key"] = config["api_key"]
            else:
                # 로컬 provider용 더미 키
                client_kwargs["api_key"] = "not-needed"
            
            if config["base_url"]:
                client_kwargs["base_url"] = config["base_url"]
            
            self._client = OpenAI(**client_kwargs)
            self._current_config = config
        
        return self._client
    
    @property
    def model(self) -> str:
        """현재 사용할 모델명"""
        return self._get_config()["model"]
    
    @property
    def provider(self) -> str:
        """현재 provider ID"""
        return self._get_config()["provider"]
    
    @property
    def is_available(self) -> bool:
        """클라이언트 사용 가능 여부"""
        config = self._get_config()
        # 로컬 provider는 API 키 불필요
        if config["provider"] in ["ollama", "lmstudio"]:
            return bool(config["base_url"])
        return bool(config["api_key"])
    
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Optional[str]:
        """
        채팅 완성 요청
        
        Args:
            messages: 메시지 목록 [{"role": "system/user/assistant", "content": "..."}]
            temperature: 생성 온도
            max_tokens: 최대 토큰 수
        
        Returns:
            응답 텍스트 또는 None (실패 시)
        """
        if not self.client:
            return None
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM API 오류: {e}")
            return None
    
    def _image_to_base64(self, image: Union[Image.Image, bytes]) -> str:
        """이미지를 base64 문자열로 변환"""
        if isinstance(image, Image.Image):
            buffered = BytesIO()
            # RGB로 변환 (RGBA인 경우)
            if image.mode in ('RGBA', 'P'):
                image = image.convert('RGB')
            image.save(buffered, format="JPEG", quality=85)
            img_bytes = buffered.getvalue()
        else:
            img_bytes = image
        
        return base64.b64encode(img_bytes).decode('utf-8')
    
    def analyze_image(
        self,
        image: Union[Image.Image, bytes],
        instruction: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> Optional[str]:
        """
        Vision LLM을 사용하여 이미지 분석
        
        Args:
            image: PIL Image 또는 이미지 바이트
            instruction: 분석 지시사항
            temperature: 생성 온도
            max_tokens: 최대 토큰 수
        
        Returns:
            분석 결과 텍스트 또는 None (실패 시)
        """
        if not self.client:
            return None
        
        try:
            base64_image = self._image_to_base64(image)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": instruction
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Vision LLM API 오류: {e}")
            return None
    
    def invalidate(self) -> None:
        """클라이언트 캐시 무효화 (설정 변경 시 호출)"""
        self._client = None
        self._current_config = {}


# 전역 LLM 클라이언트 인스턴스
llm_client = LLMClient()
