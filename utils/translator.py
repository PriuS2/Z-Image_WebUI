"""OpenAI 호환 API를 사용한 번역 유틸리티"""

import re
from typing import Optional

from utils.llm_client import llm_client


class Translator:
    """한국어 프롬프트를 영어로 번역 (다양한 OpenAI 호환 provider 지원)"""
    
    DEFAULT_SYSTEM_PROMPT = """You are a professional translator specializing in image generation prompts.
Your task is to translate Korean prompts to English, optimizing them for AI image generation.

Rules:
1. Translate accurately while preserving the artistic intent
2. Keep technical terms (like "8k", "HD", etc.) as-is
3. Maintain any special formatting like brackets [], parentheses (), or colons :
4. If the input is already in English, return it unchanged
5. Do not add any explanation, just output the translated prompt
6. Preserve any weight syntax like (word:1.5) exactly as written"""

    def __init__(self, api_key: Optional[str] = None):
        # 레거시 호환성 유지
        self.api_key = api_key
    
    @property
    def system_prompt(self) -> str:
        """설정에서 시스템 프롬프트 가져오기 (없으면 기본값 사용)"""
        from utils.settings import settings
        return settings.get("translate_system_prompt") or self.DEFAULT_SYSTEM_PROMPT
    
    def set_api_key(self, api_key: str) -> None:
        """API 키 설정 (레거시 호환)"""
        self.api_key = api_key
        # 새 시스템은 settings를 통해 관리됨
    
    def is_korean(self, text: str) -> bool:
        """텍스트에 한국어가 포함되어 있는지 확인"""
        korean_pattern = re.compile('[가-힣]')
        return bool(korean_pattern.search(text))
    
    def translate(self, text: str, target_lang: str = "en") -> tuple[str, bool]:
        """
        텍스트 번역
        
        Args:
            text: 번역할 텍스트
            target_lang: 대상 언어 (기본: 영어)
        
        Returns:
            tuple: (번역된 텍스트, 성공 여부)
        """
        if not text.strip():
            return text, True
        
        # 이미 영어인 경우 그대로 반환
        if not self.is_korean(text):
            return text, True
        
        if not llm_client.is_available:
            return text, False
        
        try:
            result = llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Translate to English:\n{text}"}
                ],
                temperature=0.3,
                max_tokens=1000,
            )
            
            if result:
                return result, True
            return text, False
        except Exception as e:
            print(f"번역 오류: {e}")
            return text, False
    
    def reverse_translate(self, text: str) -> tuple[str, bool]:
        """
        영어 프롬프트를 한국어로 역번역
        
        Args:
            text: 영어 텍스트
        
        Returns:
            tuple: (한국어 번역, 성공 여부)
        """
        if not text.strip():
            return text, True
        
        # 이미 한국어인 경우 그대로 반환
        if self.is_korean(text):
            return text, True
        
        if not llm_client.is_available:
            return text, False
        
        try:
            result = llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": """You are a professional translator.
Translate the given English image generation prompt to natural Korean.
Keep technical terms like "8k", "HD", style names, etc. as-is.
Maintain special formatting like brackets [], parentheses (), weight syntax (word:1.5).
Do not add any explanation, just output the translated prompt in Korean."""},
                    {"role": "user", "content": f"Translate to Korean:\n{text}"}
                ],
                temperature=0.3,
                max_tokens=1000,
            )
            
            if result:
                return result, True
            return text, False
        except Exception as e:
            print(f"역번역 오류: {e}")
            return text, False


# 전역 번역기 인스턴스
translator = Translator()
