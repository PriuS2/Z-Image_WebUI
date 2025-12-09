"""설정 관리 유틸리티"""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

from config.defaults import (
    DATA_DIR,
    OUTPUTS_DIR,
    MODELS_DIR,
    DEFAULT_GENERATION_SETTINGS,
    THEMES,
)


# 지원하는 LLM Provider 목록
LLM_PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
    },
    "openrouter": {
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "openai/gpt-4o-mini",
        "models": ["openai/gpt-4o-mini", "openai/gpt-4o", "anthropic/claude-3.5-sonnet", "google/gemini-pro-1.5", "meta-llama/llama-3.1-70b-instruct"],
    },
    "groq": {
        "name": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.1-70b-versatile",
        "models": ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
    },
    "together": {
        "name": "Together AI",
        "base_url": "https://api.together.xyz/v1",
        "default_model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "models": ["meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", "mistralai/Mixtral-8x7B-Instruct-v0.1"],
    },
    "ollama": {
        "name": "Ollama (로컬)",
        "base_url": "http://localhost:11434/v1",
        "default_model": "llama3.1",
        "models": ["llama3.1", "llama3.1:70b", "mistral", "mixtral", "qwen2.5"],
    },
    "lmstudio": {
        "name": "LM Studio (로컬)",
        "base_url": "http://localhost:1234/v1",
        "default_model": "local-model",
        "models": ["local-model"],
    },
    "custom": {
        "name": "커스텀",
        "base_url": "",
        "default_model": "",
        "models": [],
    },
}


class SettingsManager:
    """애플리케이션 설정 관리"""
    
    def __init__(self, settings_file: Optional[Path] = None):
        self.settings_file = settings_file or DATA_DIR / "settings.yaml"
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)
        self._settings = self._load_defaults()
        self._load()
    
    def _load_defaults(self) -> Dict[str, Any]:
        """기본 설정값 로드"""
        return {
            # API 설정 (레거시 호환)
            "openai_api_key": "",
            
            # LLM Provider 설정
            "llm_provider": "openai",
            "llm_api_key": "",
            "llm_base_url": "",  # 커스텀 provider용
            "llm_model": "",  # 비어있으면 provider 기본값 사용
            
            # 경로 설정
            "model_path": str(MODELS_DIR),
            "output_path": str(OUTPUTS_DIR),
            
            # 생성 설정
            "generation": DEFAULT_GENERATION_SETTINGS.copy(),
            
            # 모델 설정
            "quantization": "BF16 (기본, 최고품질)",
            "cpu_offload": False,
            
            # UI 설정
            "theme": "다크 모드",
            "language": "ko",
            
            # 파일명 설정
            "filename_pattern": "{date}_{time}_{seed}",
            
            # 시스템 프롬프트 (번역/향상)
            "translate_system_prompt": """You are a professional translator specializing in image generation prompts.
Your task is to translate Korean prompts to English, optimizing them for AI image generation.

Rules:
1. Translate accurately while preserving the artistic intent
2. Keep technical terms (like "8k", "HD", etc.) as-is
3. Maintain any special formatting like brackets [], parentheses (), or colons :
4. If the input is already in English, return it unchanged
5. Do not add any explanation, just output the translated prompt
6. Preserve any weight syntax like (word:1.5) exactly as written""",
            
            "enhance_system_prompt": """You are an expert at writing prompts for AI image generation based on the Imagen prompt guide.

Your task is to enhance the given prompt following this structure:
1. **Subject**: The main object, person, animal, or scene
2. **Context/Background**: Where the subject is placed (studio, outdoor, indoor, etc.)
3. **Style**: Art style (photography, painting, sketch, digital art, etc.)

Enhancement Guidelines:
- Use descriptive language with detailed adjectives and adverbs
- For photography: Add camera settings (lens type: 35mm/50mm/macro/wide-angle, lighting: natural/dramatic/warm/cold, film type: black-and-white/polaroid, focus: bokeh/soft focus/motion blur)
- For art: Specify technique (pencil drawing, charcoal, pastel painting, digital art, watercolor)
- Add quality modifiers: "high quality", "4K", "HDR", "professional", "detailed", "sharp focus"
- Reference specific art movements if appropriate (impressionism, pop art, art deco, renaissance)
- For portraits: Mention "portrait", camera proximity (close-up, medium shot)
- For landscapes: Use wide-angle lens references, mention lighting conditions (golden hour, dramatic sky)

Rules:
1. Keep the original intent and subject matter
2. Structure: Subject → Details → Style → Quality modifiers
3. Keep it concise but comprehensive (max 150 words)
4. Output ONLY the enhanced prompt, no explanations
5. Preserve any weight syntax like (word:1.5) if present""",
            
            # 기타
            "last_used": None,
        }
    
    def _load(self) -> None:
        """설정 파일에서 로드"""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    saved = yaml.safe_load(f) or {}
                    # 기존 설정에 저장된 값 병합
                    self._merge_settings(self._settings, saved)
            except Exception as e:
                print(f"설정 로드 실패: {e}")
    
    def _merge_settings(self, base: dict, updates: dict) -> None:
        """설정 병합 (중첩 딕셔너리 지원)"""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_settings(base[key], value)
            else:
                base[key] = value
    
    def save(self) -> None:
        """설정 파일에 저장"""
        self._settings["last_used"] = datetime.now().isoformat()
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                yaml.dump(self._settings, f, allow_unicode=True, default_flow_style=False)
        except Exception as e:
            print(f"설정 저장 실패: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """설정값 가져오기 (점 표기법 지원: 'generation.width')"""
        keys = key.split('.')
        value = self._settings
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """설정값 설정 (점 표기법 지원)"""
        keys = key.split('.')
        target = self._settings
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value
        self.save()
    
    def get_all(self) -> Dict[str, Any]:
        """모든 설정 가져오기"""
        return self._settings.copy()
    
    def export_to_json(self, filepath: Path) -> None:
        """설정을 JSON 파일로 내보내기"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self._settings, f, indent=2, ensure_ascii=False)
    
    def import_from_json(self, filepath: Path) -> bool:
        """JSON 파일에서 설정 가져오기"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                imported = json.load(f)
                self._merge_settings(self._settings, imported)
                self.save()
                return True
        except Exception as e:
            print(f"설정 가져오기 실패: {e}")
            return False
    
    def reset_to_defaults(self) -> None:
        """기본값으로 초기화"""
        self._settings = self._load_defaults()
        self.save()


# 전역 설정 인스턴스
settings = SettingsManager()
