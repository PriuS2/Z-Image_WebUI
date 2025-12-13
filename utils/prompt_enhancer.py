"""AI 프롬프트 향상 유틸리티"""

from typing import Optional

from utils.llm_client import llm_client


class PromptEnhancer:
    """AI를 사용한 프롬프트 향상 (다양한 OpenAI 호환 provider 지원)"""
    
    DEFAULT_SYSTEM_PROMPT = """You are an expert prompt writer for AI image generation.

Your job: rewrite the user's prompt into a single, vivid, natural-language prompt that preserves the original intent, but adds concrete, useful details that improve results.

Write style:
- Use complete sentences and natural language (no tag lists like "a, b, c").
- Be detailed. Target length: 100–300 words (optimal). Never output a very short prompt under 50 words unless the user explicitly asks for brevity.
- Keep instructions internally consistent; do not add conflicting details.

What to add (when helpful and compatible with the user's intent):
- Subject: who/what is shown, key features, action/pose.
- Scene & context: environment, time of day, weather, setting details, background elements.
- Lighting & mood: direction/quality of light (soft window light, warm pendant lights, golden hour), atmosphere, emotional tone.
- Materials & colors: textures, fabrics, surfaces, color palette.
- Composition & camera: shot type (close-up/medium/wide), camera angle, lens (e.g., 24mm/35mm/50mm/macro), depth of field (bokeh/shallow DOF), photographic style (editorial, architectural, travel, product).

Language:
- Output the enhanced prompt in English or Chinese (or both) depending on the user's input and intent.
- If the user provides Chinese text or asks for bilingual signage, include Chinese exactly and keep layout instructions clear.

Text rendering (important):
- If the user includes specific text (in quotes, or explicit strings), keep it EXACTLY unchanged.
- Describe placement, typography (font style), color, illumination, and how it integrates with the scene.
- For bilingual signage, include BOTH strings exactly and specify layout (e.g., English above, Chinese below).

Hard rules:
- Output ONLY the final enhanced prompt (no explanations, no headings, no lists).
- Do NOT add negative prompts.
- Preserve any weight syntax like (word:1.5) and any bracketed/parenthesized formatting exactly as written."""

    ENHANCEMENT_STYLES = {
        "기본": "Enhance this prompt for better AI image generation:",
        "상세하게": "Make this prompt much more detailed and descriptive for photorealistic results:",
        "간결하게": "Enhance this prompt while keeping it concise and focused:",
        "예술적": "Transform this prompt into a more artistic and creative version:",
        "사실적": "Enhance this prompt for maximum photorealism and detail:",
        "애니메이션": "Enhance this prompt for anime/illustration style output:",
        # Imagen 가이드 기반 스타일
        "인물사진": "Enhance this prompt for portrait photography. Add: 35mm lens, portrait style, close-up, natural lighting, bokeh, depth of field:",
        "풍경사진": "Enhance this prompt for landscape photography. Add: wide-angle 10-24mm lens, golden hour lighting, sharp focus, HDR:",
        "제품/음식": "Enhance this prompt for product/food photography. Add: macro lens 60-105mm, studio lighting, high precision, controlled lighting:",
        "모션/스포츠": "Enhance this prompt for action/sports photography. Add: telephoto 100-400mm lens, fast shutter speed, motion tracking:",
        "스케치/드로잉": "Transform this prompt into a sketch or drawing style. Add: pencil drawing, charcoal, detailed linework:",
        "유화/파스텔": "Transform this prompt into a painting style. Add: oil painting, pastel colors, artistic brushstrokes:",
        "디지털아트": "Transform this prompt into digital art style. Add: digital art, vibrant colors, detailed, professional:",
        "빈티지/레트로": "Transform this prompt into vintage/retro style. Add: film noir, polaroid, black-and-white, vintage tones:",
    }

    def __init__(self, api_key: Optional[str] = None):
        # 레거시 호환성 유지
        self.api_key = api_key
    
    @property
    def system_prompt(self) -> str:
        """설정에서 시스템 프롬프트 가져오기 (없으면 기본값 사용)"""
        from utils.settings import settings
        return settings.get("enhance_system_prompt") or self.DEFAULT_SYSTEM_PROMPT
    
    def set_api_key(self, api_key: str) -> None:
        """API 키 설정 (레거시 호환)"""
        self.api_key = api_key
        # 새 시스템은 settings를 통해 관리됨
    
    def enhance(
        self, 
        prompt: str, 
        style: str = "기본",
        custom_instruction: Optional[str] = None
    ) -> tuple[str, bool]:
        """
        프롬프트 향상
        
        Args:
            prompt: 원본 프롬프트
            style: 향상 스타일
            custom_instruction: 사용자 정의 지시사항
        
        Returns:
            tuple: (향상된 프롬프트, 성공 여부)
        """
        if not prompt.strip():
            return prompt, True
        
        if not llm_client.is_available:
            return prompt, False
        
        instruction = custom_instruction or self.ENHANCEMENT_STYLES.get(style, self.ENHANCEMENT_STYLES["기본"])
        
        try:
            result = llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"{instruction}\n\n{prompt}"}
                ],
                temperature=0.7,
                max_tokens=500,
            )
            
            if result:
                return result, True
            return prompt, False
        except Exception as e:
            print(f"프롬프트 향상 오류: {e}")
            return prompt, False


# 전역 인스턴스
prompt_enhancer = PromptEnhancer()
