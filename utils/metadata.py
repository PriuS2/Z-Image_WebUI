"""이미지 메타데이터 처리 유틸리티"""

import json
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

from PIL import Image
from PIL.PngImagePlugin import PngInfo


class ImageMetadata:
    """PNG 이미지 메타데이터 관리"""
    
    METADATA_KEY = "zimage_params"
    
    @staticmethod
    def create_metadata(
        prompt: str,
        negative_prompt: str = "",
        seed: int = -1,
        width: int = 512,
        height: int = 512,
        steps: int = 8,
        guidance_scale: float = 0.0,
        model: str = "",
        quantization: str = "",
        **extra_params
    ) -> Dict[str, Any]:
        """메타데이터 딕셔너리 생성"""
        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "model": model,
            "quantization": quantization,
            "generated_at": datetime.now().isoformat(),
            "generator": "Z-Image WebUI",
            **extra_params
        }
    
    @staticmethod
    def embed_metadata(image: Image.Image, metadata: Dict[str, Any]) -> Image.Image:
        """이미지에 메타데이터 임베딩"""
        png_info = PngInfo()
        png_info.add_text(ImageMetadata.METADATA_KEY, json.dumps(metadata, ensure_ascii=False))
        
        # 메타데이터가 포함된 새 이미지 반환
        image.info[ImageMetadata.METADATA_KEY] = json.dumps(metadata, ensure_ascii=False)
        return image
    
    @staticmethod
    def save_with_metadata(
        image: Image.Image, 
        filepath: Path, 
        metadata: Dict[str, Any]
    ) -> None:
        """메타데이터와 함께 이미지 저장"""
        png_info = PngInfo()
        png_info.add_text(ImageMetadata.METADATA_KEY, json.dumps(metadata, ensure_ascii=False))
        image.save(filepath, pnginfo=png_info)
    
    @staticmethod
    def read_metadata(filepath: Path) -> Optional[Dict[str, Any]]:
        """이미지에서 메타데이터 읽기"""
        try:
            with Image.open(filepath) as img:
                if ImageMetadata.METADATA_KEY in img.info:
                    return json.loads(img.info[ImageMetadata.METADATA_KEY])
                # 다른 형식의 메타데이터도 시도
                if "parameters" in img.info:
                    return ImageMetadata._parse_a1111_metadata(img.info["parameters"])
        except Exception as e:
            print(f"메타데이터 읽기 오류: {e}")
        return None
    
    @staticmethod
    def _parse_a1111_metadata(params_str: str) -> Dict[str, Any]:
        """A1111 형식의 메타데이터 파싱 (호환성)"""
        result = {"raw_parameters": params_str}
        try:
            # 간단한 파싱 시도
            lines = params_str.split('\n')
            if lines:
                result["prompt"] = lines[0]
        except:
            pass
        return result
    
    @staticmethod
    def format_for_display(metadata: Dict[str, Any]) -> str:
        """메타데이터를 읽기 좋은 형식으로 변환"""
        if not metadata:
            return "메타데이터 없음"
        
        lines = []
        if "prompt" in metadata:
            lines.append(f"📝 프롬프트: {metadata['prompt']}")
        if "negative_prompt" in metadata and metadata["negative_prompt"]:
            lines.append(f"🚫 네거티브: {metadata['negative_prompt']}")
        if "seed" in metadata:
            lines.append(f"🎲 시드: {metadata['seed']}")
        if "width" in metadata and "height" in metadata:
            lines.append(f"📐 해상도: {metadata['width']}x{metadata['height']}")
        if "steps" in metadata:
            lines.append(f"🔄 스텝: {metadata['steps']}")
        if "model" in metadata:
            lines.append(f"🤖 모델: {metadata['model']}")
        if "generated_at" in metadata:
            lines.append(f"📅 생성일: {metadata['generated_at']}")
        
        return "\n".join(lines)


# 파일명 생성 유틸리티
class FilenameGenerator:
    """자동 파일명 생성"""
    
    def __init__(self):
        self._counter = 0
    
    def generate(
        self,
        pattern: str,
        prompt: str = "",
        seed: int = 0,
        extension: str = ".png"
    ) -> str:
        """
        패턴에 따라 파일명 생성
        
        패턴 변수:
        - {date}: YYYYMMDD
        - {time}: HHMMSS
        - {seed}: 시드 값
        - {prompt_short}: 프롬프트 앞 30자
        - {counter}: 순차 번호
        """
        now = datetime.now()
        self._counter += 1
        
        # 프롬프트 정리 (파일명에 사용 불가능한 문자 제거)
        prompt_short = prompt[:30] if prompt else "image"
        prompt_short = "".join(c if c.isalnum() or c in "_ -" else "_" for c in prompt_short)
        prompt_short = prompt_short.strip("_- ")
        
        filename = pattern.format(
            date=now.strftime("%Y%m%d"),
            time=now.strftime("%H%M%S"),
            seed=seed,
            prompt_short=prompt_short or "image",
            counter=f"{self._counter:04d}"
        )
        
        return filename + extension
    
    def reset_counter(self) -> None:
        """카운터 초기화"""
        self._counter = 0


# 전역 인스턴스
filename_generator = FilenameGenerator()
