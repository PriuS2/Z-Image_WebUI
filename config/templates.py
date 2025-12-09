"""프롬프트 템플릿 및 프리셋"""

# 프롬프트 템플릿
PROMPT_TEMPLATES = {
    "📷 인물 사진 (포트레이트)": {
        "prompt": "A professional portrait photograph of {subject}, {style} lighting, {background} background, high resolution, detailed skin texture, sharp focus, 8k quality",
        "negative": "blurry, low quality, distorted face, bad anatomy, extra limbs",
        "variables": {
            "subject": "a young woman with natural makeup",
            "style": "soft studio",
            "background": "neutral gray"
        }
    },
    "🎨 애니메이션 스타일": {
        "prompt": "{subject}, anime style, {mood} atmosphere, {detail} details, vibrant colors, clean lines, high quality anime artwork",
        "negative": "realistic, photo, 3d render, blurry, low quality",
        "variables": {
            "subject": "a beautiful anime girl",
            "mood": "peaceful",
            "detail": "intricate"
        }
    },
    "🏞️ 풍경": {
        "prompt": "A stunning {landscape_type} landscape, {time_of_day}, {weather} weather, {style} photography style, ultra high resolution, cinematic composition",
        "negative": "people, text, watermark, blurry, low quality",
        "variables": {
            "landscape_type": "mountain",
            "time_of_day": "golden hour sunset",
            "weather": "clear",
            "style": "professional"
        }
    },
    "🛍️ 제품 사진": {
        "prompt": "[Product: {product}] Floating centered against pure white background (hex #FFFFFF), soft drop shadow, {angle} view, studio lighting setup, clear details visibility, commercial photography style, high resolution",
        "negative": "cluttered background, shadows, low quality, blurry",
        "variables": {
            "product": "luxury watch",
            "angle": "45-degree"
        }
    },
    "🎭 판타지 아트": {
        "prompt": "Epic fantasy artwork of {subject}, {setting}, magical atmosphere, {lighting} lighting, highly detailed, concept art style, trending on artstation",
        "negative": "modern elements, low quality, blurry, amateur",
        "variables": {
            "subject": "a powerful wizard",
            "setting": "ancient mystical forest",
            "lighting": "dramatic volumetric"
        }
    },
    "🔮 사이버펑크": {
        "prompt": "{subject} in a cyberpunk cityscape, neon lights, rain-slicked streets, futuristic technology, {mood} atmosphere, cinematic lighting, highly detailed, 8k",
        "negative": "natural setting, daylight, low tech, blurry, low quality",
        "variables": {
            "subject": "a cybernetic enhanced human",
            "mood": "dystopian noir"
        }
    },
    "📚 웹소설 표지": {
        "prompt": "A high-quality digital illustration in the style of a Korean webnovel cover. {character_description}, {scene_description}, {art_style}, dramatic lighting, high contrast",
        "negative": "low quality, blurry, amateur, western comic style",
        "variables": {
            "character_description": "A handsome young man in traditional robes",
            "scene_description": "standing in a grand palace hall",
            "art_style": "semi-realistic anime style"
        }
    },
    "🖼️ 미니멀 아트": {
        "prompt": "Minimalist {subject}, clean composition, {color_scheme} color palette, simple shapes, modern design, high contrast, artistic",
        "negative": "cluttered, complex, realistic, noisy, textured",
        "variables": {
            "subject": "geometric abstract art",
            "color_scheme": "monochrome"
        }
    },
}

# 스타일 수정자 (프롬프트에 추가 가능)
STYLE_MODIFIERS = {
    "품질 향상": [
        "masterpiece",
        "best quality",
        "ultra detailed",
        "8k resolution",
        "high resolution",
        "sharp focus",
        "professional",
    ],
    "조명": [
        "studio lighting",
        "natural lighting",
        "dramatic lighting",
        "soft lighting",
        "cinematic lighting",
        "golden hour",
        "backlit",
        "rim lighting",
    ],
    "스타일": [
        "photorealistic",
        "hyperrealistic",
        "anime style",
        "oil painting",
        "watercolor",
        "digital art",
        "concept art",
        "illustration",
    ],
    "분위기": [
        "peaceful",
        "dramatic",
        "mysterious",
        "vibrant",
        "melancholic",
        "ethereal",
        "epic",
        "cozy",
    ],
}

# 네거티브 프롬프트 프리셋
NEGATIVE_PRESETS = {
    "기본": "low quality, blurry, distorted",
    "인물 사진": "blurry, low quality, distorted face, bad anatomy, extra limbs, missing limbs, fused fingers, too many fingers, long neck",
    "애니메이션": "realistic, photo, 3d render, blurry, low quality, bad anatomy, extra limbs",
    "풍경": "people, text, watermark, blurry, low quality, oversaturated",
    "제품": "cluttered background, shadows, low quality, blurry, text, watermark",
    "없음": "",
}
