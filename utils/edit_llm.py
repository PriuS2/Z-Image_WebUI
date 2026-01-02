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
    
    def translate_negative(self, text: str) -> Tuple[str, bool]:
        """
        Negative prompt 번역 (한국어 → 영어)
        Negative prompt는 제외할 요소를 나열하는 형태로, 일반 편집 지시어와 다름
        
        Args:
            text: 번역할 negative prompt
        
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
                    {"role": "system", "content": """You are a translator for AI image editing negative prompts.
Translate Korean negative prompts to English.

Negative prompts describe unwanted elements to avoid in image editing.
Common patterns include: blurry, distorted, low quality, watermark, etc.

Rules:
1. Translate each term/phrase accurately
2. Keep comma-separated format if present
3. If already in English, return unchanged
4. Output ONLY the translated negative prompt, no explanation

Examples:
- "흐릿함, 저화질" → "blurry, low quality"
- "워터마크, 텍스트 왜곡" → "watermark, distorted text"
- "노이즈, 픽셀화" → "noise, pixelated"
"""},
                    {"role": "user", "content": f"Translate negative prompt:\n{text}"}
                ],
                temperature=0.3,
                max_tokens=300,
            )
            
            if result:
                return result.strip(), True
            return text, False
        except Exception as e:
            print(f"Negative prompt 번역 오류: {e}")
            return text, False


class EditEnhancer:
    """편집 지시어 향상 (간단한 지시어 → 상세한 지시어)"""
    
    DEFAULT_SYSTEM_PROMPT = """You are an expert at writing effective image editing instructions for AI image editing models.

Your job: rewrite the user's edit instruction into ONE clear, step-by-step friendly instruction that performs a single edit action at a time (one change per run).

Core principles:
- Preserve the original intent and keep everything else unchanged unless explicitly requested (identity, pose, composition, lighting, perspective).
- Use natural language (no tag lists).
- Structure the instruction as: Action → Target → Location/Scope → Desired details (materials/colors/style) → What must remain unchanged.
- Be specific: name the object, where it is, and what exactly changes (color shade, texture, size, style, placement).
- Avoid conflicting instructions. Do NOT add negative prompts.

One-change rule:
- If the user's input contains multiple edits, output only ONE instruction for the first/most direct edit, suitable for iterative editing ("one thing at a time").

Special handling for text rendering (critical):
- If the edit involves adding/changing any visible text in the image, you MUST wrap the exact target text in quotation marks: '...' or \"...\" (also supports Chinese quotes ‘...’ / “...”).
- Keep any quoted text EXACTLY unchanged (characters, spacing, punctuation). If the user provides unquoted text to render, add quotes around it.
- Describe placement, typography (font style), color, and illumination so the text reads clearly.

Output rules:
- Output ONLY the enhanced editing instruction (no explanations, no headings, no lists).
- Keep it concise but sufficiently specific (aim ~20–60 words)."""

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
    
    DEFAULT_SYSTEM_PROMPT = """You are a creative, practical image editing assistant.
Based on the user's context and/or an image description, suggest 3–5 editing ideas that are easy to execute iteratively.

Guidelines:
1. Each suggestion must be a SINGLE edit (one change per run). Do not bundle multiple changes in one line.
2. Keep the rest unchanged unless the user implies otherwise (identity, pose, composition, lighting).
3. Suggest a variety of edit types across the list (e.g., one lighting idea, one background idea, one add/remove object idea, one style idea, one color/tone idea) when possible.
4. Keep each suggestion concise (10–15 words max), in natural language (no tag lists).
5. Avoid conflicting instructions. Do NOT use negative prompts.

Special handling for text rendering:
- If you suggest adding or changing any visible text, ALWAYS put the exact text in quotes: '...' or \"...\" (also supports Chinese quotes ‘...’ / “...”).
- Keep quoted text exact; do not paraphrase it.

Output format:
- Output ONLY a numbered list (1–5). No extra commentary."""

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


