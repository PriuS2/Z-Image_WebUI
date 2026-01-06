"""설정 API 라우터"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config.defaults import (
    QUANTIZATION_OPTIONS,
    RESOLUTION_PRESETS,
    OUTPUTS_DIR,
    QWEN_EDIT_AUTO_UNLOAD_TIMEOUT,
)
from utils.settings import settings
from utils.session import is_localhost
from utils.translator import translator, Translator
from utils.prompt_enhancer import prompt_enhancer, PromptEnhancer

from routers.dependencies import (
    get_session_from_request,
    set_session_cookie,
)


router = APIRouter()


# ============= Pydantic 모델 =============
class SettingsRequest(BaseModel):
    openai_api_key: str = ""  # 레거시 호환
    output_path: str = ""
    filename_pattern: str = "{date}_{time}_{seed}"
    # LLM Provider 설정
    llm_provider: str = ""
    llm_api_key: str = ""
    # Optional 필드 (None이 아닐 때만 저장)
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None
    # 시스템 프롬프트 (번역/향상)
    translate_system_prompt: Optional[str] = None
    enhance_system_prompt: Optional[str] = None
    # 자동 언로드 설정
    auto_unload_enabled: Optional[bool] = None
    auto_unload_timeout: Optional[int] = None
    # 편집 모델 자동 언로드 설정
    edit_auto_unload_enabled: Optional[bool] = None
    edit_auto_unload_timeout: Optional[int] = None
    # 모델 설정 (관리자 전용)
    quantization: Optional[str] = None
    cpu_offload: Optional[bool] = None
    # 편집 모델 설정 (관리자 전용) - Qwen은 4bit NF4 고정
    edit_cpu_offload: Optional[bool] = None


class SystemPromptsRequest(BaseModel):
    translate_system_prompt: Optional[str] = None
    enhance_system_prompt: Optional[str] = None
    # 편집 시스템 프롬프트 (개인화)
    edit_translate_system_prompt: Optional[str] = None
    edit_enhance_system_prompt: Optional[str] = None
    edit_suggest_system_prompt: Optional[str] = None


# ============= 엔드포인트 =============
@router.post("/settings")
async def save_settings(request: Request, settings_request: SettingsRequest):
    """설정 저장 (localhost만 허용)"""
    # localhost 체크
    client_host = request.client.host if request.client else None
    if not is_localhost(client_host):
        raise HTTPException(403, "설정 변경은 localhost에서만 가능합니다.")
    
    from utils.llm_client import llm_client
    
    # 레거시 호환
    if settings_request.openai_api_key:
        settings.set("openai_api_key", settings_request.openai_api_key)
        translator.set_api_key(settings_request.openai_api_key)
        prompt_enhancer.set_api_key(settings_request.openai_api_key)
    
    # LLM Provider 설정
    if settings_request.llm_provider:
        settings.set("llm_provider", settings_request.llm_provider)
    
    if settings_request.llm_api_key:
        settings.set("llm_api_key", settings_request.llm_api_key)
        settings.set("openai_api_key", settings_request.llm_api_key)
    
    if settings_request.llm_base_url is not None:
        settings.set("llm_base_url", settings_request.llm_base_url)
    
    if settings_request.llm_model is not None:
        settings.set("llm_model", settings_request.llm_model)
    
    llm_client.invalidate()
    
    if settings_request.output_path:
        settings.set("output_path", settings_request.output_path)
    
    if settings_request.filename_pattern:
        settings.set("filename_pattern", settings_request.filename_pattern)
    
    if settings_request.translate_system_prompt is not None:
        settings.set("translate_system_prompt", settings_request.translate_system_prompt)
    
    if settings_request.enhance_system_prompt is not None:
        settings.set("enhance_system_prompt", settings_request.enhance_system_prompt)
    
    if settings_request.auto_unload_enabled is not None:
        settings.set("auto_unload_enabled", settings_request.auto_unload_enabled)
    
    if settings_request.auto_unload_timeout is not None:
        timeout = max(1, min(1440, settings_request.auto_unload_timeout))
        settings.set("auto_unload_timeout", timeout)
    
    if settings_request.edit_auto_unload_enabled is not None:
        settings.set("edit_auto_unload_enabled", settings_request.edit_auto_unload_enabled)
    
    if settings_request.edit_auto_unload_timeout is not None:
        timeout = max(1, min(1440, settings_request.edit_auto_unload_timeout))
        settings.set("edit_auto_unload_timeout", timeout)

    # 모델 설정 (관리자 전용)
    if settings_request.quantization is not None:
        if settings_request.quantization not in QUANTIZATION_OPTIONS:
            raise HTTPException(400, f"지원하지 않는 양자화: {settings_request.quantization}")
        settings.set("quantization", settings_request.quantization)

    if settings_request.cpu_offload is not None:
        settings.set("cpu_offload", bool(settings_request.cpu_offload))

    # 편집 모델 설정 (관리자 전용) - Qwen은 4bit NF4 고정
    if settings_request.edit_cpu_offload is not None:
        settings.set("edit_cpu_offload", bool(settings_request.edit_cpu_offload))
    
    return {"success": True}


@router.get("/settings")
async def get_settings(request: Request):
    """설정 가져오기"""
    from utils.settings import LLM_PROVIDERS
    from utils.translator import Translator
    from utils.prompt_enhancer import PromptEnhancer
    from utils.edit_llm import EditTranslator, EditEnhancer, EditSuggester
    
    session = await get_session_from_request(request)
    client_host = request.client.host if request.client else None
    is_admin = is_localhost(client_host)
    
    # 세션별 시스템 프롬프트 (개인화)
    session_translate_prompt = session.get_setting("translate_system_prompt") if session else None
    session_enhance_prompt = session.get_setting("enhance_system_prompt") if session else None
    
    # 세션에 설정이 없으면 전역 설정 사용, 전역도 없으면 기본값
    translate_prompt = session_translate_prompt or settings.get("translate_system_prompt") or Translator.DEFAULT_SYSTEM_PROMPT
    enhance_prompt = session_enhance_prompt or settings.get("enhance_system_prompt") or PromptEnhancer.DEFAULT_SYSTEM_PROMPT
    
    # 편집 시스템 프롬프트 (세션별 개인화)
    session_edit_translate = session.get_setting("edit_translate_system_prompt") if session else None
    session_edit_enhance = session.get_setting("edit_enhance_system_prompt") if session else None
    session_edit_suggest = session.get_setting("edit_suggest_system_prompt") if session else None
    
    edit_translate_prompt = session_edit_translate or settings.get("edit_translate_system_prompt") or EditTranslator.DEFAULT_SYSTEM_PROMPT
    edit_enhance_prompt = session_edit_enhance or settings.get("edit_enhance_system_prompt") or EditEnhancer.DEFAULT_SYSTEM_PROMPT
    edit_suggest_prompt = session_edit_suggest or settings.get("edit_suggest_system_prompt") or EditSuggester.DEFAULT_SYSTEM_PROMPT
    
    return {
        # 관리자 여부
        "is_admin": is_admin,
        # 레거시 호환
        "openai_api_key": "***" if settings.get("openai_api_key") else "",
        # LLM Provider 설정
        "llm_provider": settings.get("llm_provider", "env"),
        "llm_api_key": "***" if settings.get("llm_api_key") else "",
        "llm_base_url": settings.get("llm_base_url", ""),
        "llm_model": settings.get("llm_model", ""),
        "llm_providers": {
            pid: {
                "name": pinfo["name"],
                "base_url": pinfo["base_url"],
                "default_model": pinfo["default_model"],
                "models": pinfo["models"],
            }
            for pid, pinfo in LLM_PROVIDERS.items()
        },
        # 시스템 프롬프트 (세션별 개인화)
        "translate_system_prompt": translate_prompt,
        "enhance_system_prompt": enhance_prompt,
        "default_translate_system_prompt": Translator.DEFAULT_SYSTEM_PROMPT,
        "default_enhance_system_prompt": PromptEnhancer.DEFAULT_SYSTEM_PROMPT,
        # 편집 시스템 프롬프트 (세션별 개인화)
        "edit_translate_system_prompt": edit_translate_prompt,
        "edit_enhance_system_prompt": edit_enhance_prompt,
        "edit_suggest_system_prompt": edit_suggest_prompt,
        "default_edit_translate_system_prompt": EditTranslator.DEFAULT_SYSTEM_PROMPT,
        "default_edit_enhance_system_prompt": EditEnhancer.DEFAULT_SYSTEM_PROMPT,
        "default_edit_suggest_system_prompt": EditSuggester.DEFAULT_SYSTEM_PROMPT,
        # 기타 설정
        "output_path": str(settings.get("output_path", OUTPUTS_DIR)),
        "filename_pattern": settings.get("filename_pattern", "{date}_{time}_{seed}"),
        # 모델 설정 (관리자만 변경 가능 - 모든 사용자에게는 현재 값만 제공)
        "quantization": settings.get("quantization", "BF16 (기본, 최고품질)"),
        "cpu_offload": settings.get("cpu_offload", False),
        "edit_quantization": settings.get("edit_quantization", "BF16 (기본, 최고품질)"),
        "edit_cpu_offload": settings.get("edit_cpu_offload", True),
        "quantization_options": list(QUANTIZATION_OPTIONS.keys()),
        "resolution_presets": RESOLUTION_PRESETS,
        # 자동 언로드 설정
        "auto_unload_enabled": settings.get("auto_unload_enabled", True),
        "auto_unload_timeout": settings.get("auto_unload_timeout", 10),
        # 편집 모델 자동 언로드 설정
        "edit_auto_unload_enabled": settings.get("edit_auto_unload_enabled", True),
        "edit_auto_unload_timeout": settings.get("edit_auto_unload_timeout", QWEN_EDIT_AUTO_UNLOAD_TIMEOUT),
    }


@router.post("/settings/prompts")
async def save_session_prompts(request: Request, prompts_request: SystemPromptsRequest):
    """시스템 프롬프트 저장 (세션별 개인화, 모든 사용자 접근 가능)"""
    session = await get_session_from_request(request)
    session_settings = session.get_settings()
    
    # 생성 시스템 프롬프트
    if prompts_request.translate_system_prompt is not None:
        if prompts_request.translate_system_prompt == '':
            # 빈 문자열이면 설정 삭제 (기본값 사용)
            session_settings.pop("translate_system_prompt", None)
        else:
            session_settings["translate_system_prompt"] = prompts_request.translate_system_prompt
    
    if prompts_request.enhance_system_prompt is not None:
        if prompts_request.enhance_system_prompt == '':
            # 빈 문자열이면 설정 삭제 (기본값 사용)
            session_settings.pop("enhance_system_prompt", None)
        else:
            session_settings["enhance_system_prompt"] = prompts_request.enhance_system_prompt
    
    # 편집 시스템 프롬프트
    if prompts_request.edit_translate_system_prompt is not None:
        if prompts_request.edit_translate_system_prompt == '':
            session_settings.pop("edit_translate_system_prompt", None)
        else:
            session_settings["edit_translate_system_prompt"] = prompts_request.edit_translate_system_prompt
    
    if prompts_request.edit_enhance_system_prompt is not None:
        if prompts_request.edit_enhance_system_prompt == '':
            session_settings.pop("edit_enhance_system_prompt", None)
        else:
            session_settings["edit_enhance_system_prompt"] = prompts_request.edit_enhance_system_prompt
    
    if prompts_request.edit_suggest_system_prompt is not None:
        if prompts_request.edit_suggest_system_prompt == '':
            session_settings.pop("edit_suggest_system_prompt", None)
        else:
            session_settings["edit_suggest_system_prompt"] = prompts_request.edit_suggest_system_prompt
    
    session.save_settings(session_settings)
    
    response = JSONResponse(content={"success": True})
    set_session_cookie(response, session)
    return response


@router.delete("/settings/prompts")
async def reset_session_prompts(request: Request):
    """시스템 프롬프트 초기화 (세션별 설정 삭제, 전역/기본값 사용)"""
    session = await get_session_from_request(request)
    
    session_settings = session.get_settings()
    # 생성 시스템 프롬프트
    if "translate_system_prompt" in session_settings:
        del session_settings["translate_system_prompt"]
    if "enhance_system_prompt" in session_settings:
        del session_settings["enhance_system_prompt"]
    # 편집 시스템 프롬프트
    if "edit_translate_system_prompt" in session_settings:
        del session_settings["edit_translate_system_prompt"]
    if "edit_enhance_system_prompt" in session_settings:
        del session_settings["edit_enhance_system_prompt"]
    if "edit_suggest_system_prompt" in session_settings:
        del session_settings["edit_suggest_system_prompt"]
    session.save_settings(session_settings)
    
    response = JSONResponse(content={"success": True})
    set_session_cookie(response, session)
    return response
