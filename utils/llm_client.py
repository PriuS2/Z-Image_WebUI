"""OpenAI 호환 API 클라이언트 유틸리티"""

from typing import Optional, Dict, Any, List
from openai import OpenAI

from utils.settings import settings, LLM_PROVIDERS


class LLMClient:
    """다양한 OpenAI 호환 provider를 지원하는 LLM 클라이언트"""
    
    def __init__(self):
        self._client: Optional[OpenAI] = None
        self._current_config: Dict[str, Any] = {}
    
    def _get_config(self) -> Dict[str, Any]:
        """현재 설정에서 LLM 구성 가져오기"""
        provider_id = settings.get("llm_provider", "openai")
        provider_info = LLM_PROVIDERS.get(provider_id, LLM_PROVIDERS["openai"])
        
        # API 키: 새 설정 우선, 레거시 호환
        api_key = settings.get("llm_api_key") or settings.get("openai_api_key", "")
        
        # Base URL: 커스텀이면 설정값, 아니면 provider 기본값
        if provider_id == "custom":
            base_url = settings.get("llm_base_url", "")
        else:
            base_url = provider_info["base_url"]
        
        # 모델: 설정값 또는 provider 기본값
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
        messages: List[Dict[str, str]],
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
    
    def invalidate(self) -> None:
        """클라이언트 캐시 무효화 (설정 변경 시 호출)"""
        self._client = None
        self._current_config = {}


# 전역 LLM 클라이언트 인스턴스
llm_client = LLMClient()
