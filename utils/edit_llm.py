"""이미지 편집용 LLM 유틸리티 (번역, 향상, 제안)"""

import re
from typing import Optional, List, Dict, Any, Tuple

from utils.llm_client import llm_client


class EditTranslator:
    """편집 지시어 번역 (한국어 → 영어)"""
    
    DEFAULT_SYSTEM_PROMPT = """You are a professional translator specializing in image editing instructions.
Your task is to translate Korean image editing instructions to English.

Rules:
1. Translate accurately while preserving the editing intent
2. Keep technical terms and style references as-is
3. If the input mentions text to render in quotes (""), keep the text inside quotes unchanged
4. If the input is already in English, return it unchanged
5. Do not add any explanation, just output the translated instruction
6. The translation should be suitable for AI image editing models

Examples:
- "고양이를 강아지로 변경" → "Change the cat to a dog"
- "배경을 해변으로 바꿔줘" → "Change the background to a beach"
- "머리 색깔을 금발로 변경" → "Change the hair color to blonde"
- "\"HELLO\" 텍스트 추가" → "Add \"HELLO\" text"
"""

    def __init__(self):
        pass
    
    @property
    def system_prompt(self) -> str:
        """설정에서 시스템 프롬프트 가져오기"""
        from utils.settings import settings
        return settings.get("edit_translate_system_prompt") or self.DEFAULT_SYSTEM_PROMPT
    
    def is_korean(self, text: str) -> bool:
        """텍스트에 한국어가 포함되어 있는지 확인"""
        korean_pattern = re.compile('[가-힣]')
        return bool(korean_pattern.search(text))
    
    def translate(self, text: str) -> Tuple[str, bool]:
        """
        편집 지시어 번역 (한국어 → 영어)
        
        Args:
            text: 번역할 텍스트
        
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
                max_tokens=500,
            )
            
            if result:
                return result.strip(), True
            return text, False
        except Exception as e:
            print(f"편집 지시어 번역 오류: {e}")
            return text, False
    
    def reverse_translate(self, text: str) -> Tuple[str, bool]:
        """
        영어 편집 지시어를 한국어로 역번역
        
        Args:
            text: 영어 텍스트
        
        Returns:
            tuple: (한국어 번역, 성공 여부)
        """
        if not text.strip():
            return text, True
        
        if self.is_korean(text):
            return text, True
        
        if not llm_client.is_available:
            return text, False
        
        try:
            result = llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": """You are a professional translator.
Translate the given English image editing instruction to natural Korean.
Keep any text in quotes unchanged.
Do not add any explanation, just output the translated instruction in Korean."""},
                    {"role": "user", "content": f"Translate to Korean:\n{text}"}
                ],
                temperature=0.3,
                max_tokens=500,
            )
            
            if result:
                return result.strip(), True
            return text, False
        except Exception as e:
            print(f"역번역 오류: {e}")
            return text, False


class EditEnhancer:
    """편집 지시어 향상 (간단한 지시어 → 상세한 지시어)"""
    
    DEFAULT_SYSTEM_PROMPT = """You are an expert at writing detailed image editing instructions for AI models.
Your task is to enhance simple editing instructions into more detailed, effective prompts.

Guidelines:
1. Keep the original intent and target of the edit
2. Add specific details about the desired result
3. If it's a style change, specify the artistic style
4. If it's a color change, be specific about the shade/tone
5. If it's adding/removing elements, describe their position and appearance
6. Keep the instruction concise but comprehensive (max 50 words)
7. Output ONLY the enhanced instruction, no explanations
8. If the instruction mentions text in quotes (""), keep it exactly as is

Examples:
- Input: "Change cat to dog"
  Output: "Transform the cat into a fluffy golden retriever dog, maintaining the same pose and position, with natural fur texture and realistic proportions"

- Input: "Make background sunset"
  Output: "Replace the background with a warm sunset scene, featuring orange and pink gradient sky, soft clouds, and golden hour lighting that naturally illuminates the subject"

- Input: "Change hair to blonde"
  Output: "Change the hair color to a natural honey blonde shade with subtle highlights, maintaining the original hairstyle and texture"
"""

    def __init__(self):
        pass
    
    @property
    def system_prompt(self) -> str:
        """설정에서 시스템 프롬프트 가져오기"""
        from utils.settings import settings
        return settings.get("edit_enhance_system_prompt") or self.DEFAULT_SYSTEM_PROMPT
    
    def enhance(self, instruction: str) -> Tuple[str, bool]:
        """
        편집 지시어 향상
        
        Args:
            instruction: 간단한 편집 지시어
        
        Returns:
            tuple: (향상된 지시어, 성공 여부)
        """
        if not instruction.strip():
            return instruction, True
        
        if not llm_client.is_available:
            return instruction, False
        
        try:
            result = llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Enhance this editing instruction:\n{instruction}"}
                ],
                temperature=0.7,
                max_tokens=200,
            )
            
            if result:
                return result.strip(), True
            return instruction, False
        except Exception as e:
            print(f"편집 지시어 향상 오류: {e}")
            return instruction, False


class EditSuggester:
    """이미지 분석 후 편집 아이디어 제안"""
    
    DEFAULT_SYSTEM_PROMPT = """You are a creative image editing assistant.
Based on the description or context provided, suggest 3-5 creative editing ideas.

Guidelines:
1. Suggest a variety of editing types: style changes, color adjustments, element additions/removals, background changes, etc.
2. Keep each suggestion concise (10-15 words max)
3. Make suggestions practical and achievable with AI image editing
4. Consider the context and purpose of the image
5. Output as a numbered list

Example output format:
1. Change the background to a tropical beach scene
2. Add dramatic sunset lighting
3. Transform the style to watercolor painting
4. Change the season to snowy winter
5. Add artistic blur to the background
"""

    def __init__(self):
        pass
    
    @property
    def system_prompt(self) -> str:
        """설정에서 시스템 프롬프트 가져오기"""
        from utils.settings import settings
        return settings.get("edit_suggest_system_prompt") or self.DEFAULT_SYSTEM_PROMPT
    
    def suggest(self, context: str = "", image_description: str = "") -> Tuple[List[str], bool]:
        """
        편집 아이디어 제안
        
        Args:
            context: 편집 맥락 또는 사용자 요청
            image_description: 이미지 설명 (있으면)
        
        Returns:
            tuple: (제안 목록, 성공 여부)
        """
        if not llm_client.is_available:
            return [], False
        
        prompt_parts = []
        if image_description:
            prompt_parts.append(f"Image description: {image_description}")
        if context:
            prompt_parts.append(f"User context: {context}")
        
        if not prompt_parts:
            prompt_parts.append("Suggest general creative image editing ideas")
        
        user_message = "\n".join(prompt_parts)
        
        try:
            result = llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.8,
                max_tokens=300,
            )
            
            if result:
                # 결과 파싱 (번호 목록 형식)
                suggestions = []
                lines = result.strip().split('\n')
                for line in lines:
                    # 번호 제거 (1. 2. 3. 등)
                    cleaned = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
                    if cleaned:
                        suggestions.append(cleaned)
                
                return suggestions[:5], True
            return [], False
        except Exception as e:
            print(f"편집 제안 오류: {e}")
            return [], False
    
    def suggest_korean(self, context: str = "", image_description: str = "") -> Tuple[List[str], bool]:
        """편집 아이디어 제안 (한국어)"""
        suggestions, success = self.suggest(context, image_description)
        
        if not success or not suggestions:
            return [], success
        
        # 한국어로 번역
        translator = EditTranslator()
        korean_suggestions = []
        
        for suggestion in suggestions:
            translated, _ = translator.reverse_translate(suggestion)
            korean_suggestions.append(translated)
        
        return korean_suggestions, True


# 전역 인스턴스
edit_translator = EditTranslator()
edit_enhancer = EditEnhancer()
edit_suggester = EditSuggester()

