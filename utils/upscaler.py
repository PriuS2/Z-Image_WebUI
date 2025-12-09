"""Real-ESRGAN 업스케일링 유틸리티"""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

# Real-ESRGAN 임포트 시도
try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    REALESRGAN_AVAILABLE = True
except ImportError:
    REALESRGAN_AVAILABLE = False
    print("Real-ESRGAN이 설치되지 않았습니다. 업스케일링 기능이 제한됩니다.")


class Upscaler:
    """Real-ESRGAN 기반 이미지 업스케일러"""
    
    # 지원하는 모델들
    MODELS = {
        "RealESRGAN_x4plus": {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "scale": 4,
            "model_class": "RRDBNet",
            "num_block": 23,
            "description": "일반 이미지용 (4x)"
        },
        "RealESRGAN_x4plus_anime_6B": {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            "scale": 4,
            "model_class": "RRDBNet",
            "num_block": 6,
            "description": "애니메이션/일러스트용 (4x)"
        },
        "RealESRGAN_x2plus": {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            "scale": 2,
            "model_class": "RRDBNet",
            "num_block": 23,
            "description": "일반 이미지용 (2x)"
        },
    }
    
    def __init__(self, model_name: str = "RealESRGAN_x4plus", device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self._upscaler = None
        self._model_path = None
        
        # 모델 저장 경로
        self.models_dir = Path.home() / ".cache" / "realesrgan"
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def is_available(self) -> bool:
        """Real-ESRGAN 사용 가능 여부"""
        return REALESRGAN_AVAILABLE
    
    def _get_model_path(self, model_name: str) -> Path:
        """모델 파일 경로 반환 (없으면 다운로드)"""
        model_info = self.MODELS.get(model_name)
        if not model_info:
            raise ValueError(f"지원하지 않는 모델: {model_name}")
        
        model_path = self.models_dir / f"{model_name}.pth"
        
        if not model_path.exists():
            print(f"모델 다운로드 중: {model_name}...")
            self._download_model(model_info["url"], model_path)
        
        return model_path
    
    def _download_model(self, url: str, save_path: Path) -> None:
        """모델 다운로드"""
        import urllib.request
        from tqdm import tqdm
        
        class DownloadProgressBar(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)
        
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=save_path.name) as t:
            urllib.request.urlretrieve(url, save_path, reporthook=t.update_to)
    
    def load_model(self, model_name: Optional[str] = None) -> bool:
        """모델 로드"""
        if not REALESRGAN_AVAILABLE:
            return False
        
        model_name = model_name or self.model_name
        model_info = self.MODELS.get(model_name)
        
        if not model_info:
            print(f"지원하지 않는 모델: {model_name}")
            return False
        
        try:
            model_path = self._get_model_path(model_name)
            
            # 모델 아키텍처 생성
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=model_info["num_block"],
                num_grow_ch=32,
                scale=model_info["scale"]
            )
            
            # 디바이스 설정
            if self.device == "auto":
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            else:
                device = self.device
            
            # 업스케일러 초기화
            self._upscaler = RealESRGANer(
                scale=model_info["scale"],
                model_path=str(model_path),
                model=model,
                tile=0,  # 타일 사이즈 (0 = 비활성화)
                tile_pad=10,
                pre_pad=0,
                half=True if device == "cuda" else False,
                device=device
            )
            
            self.model_name = model_name
            self._model_path = model_path
            print(f"업스케일러 로드 완료: {model_name}")
            return True
            
        except Exception as e:
            print(f"업스케일러 로드 실패: {e}")
            return False
    
    def upscale(
        self, 
        image: Image.Image, 
        outscale: Optional[int] = None
    ) -> Tuple[Optional[Image.Image], str]:
        """
        이미지 업스케일
        
        Args:
            image: 입력 이미지
            outscale: 출력 배율 (None이면 모델 기본값 사용)
        
        Returns:
            tuple: (업스케일된 이미지, 상태 메시지)
        """
        if not REALESRGAN_AVAILABLE:
            return None, "Real-ESRGAN이 설치되지 않았습니다."
        
        if self._upscaler is None:
            if not self.load_model():
                return None, "업스케일러 로드 실패"
        
        try:
            # PIL Image -> numpy array
            img_array = np.array(image)
            
            # BGR로 변환 (OpenCV 형식)
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = img_array[:, :, ::-1]
            
            # 업스케일 수행
            model_info = self.MODELS[self.model_name]
            scale = outscale or model_info["scale"]
            
            output, _ = self._upscaler.enhance(img_array, outscale=scale)
            
            # RGB로 다시 변환
            if len(output.shape) == 3 and output.shape[2] == 3:
                output = output[:, :, ::-1]
            
            # numpy array -> PIL Image
            result = Image.fromarray(output)
            
            return result, f"업스케일 완료 ({scale}x)"
            
        except Exception as e:
            return None, f"업스케일 오류: {e}"
    
    def upscale_with_fallback(
        self, 
        image: Image.Image, 
        scale: int = 2
    ) -> Tuple[Image.Image, str]:
        """
        업스케일 (Real-ESRGAN 실패 시 Lanczos 폴백)
        """
        if REALESRGAN_AVAILABLE and self._upscaler is not None:
            result, msg = self.upscale(image, scale)
            if result is not None:
                return result, msg
        
        # Lanczos 폴백
        new_size = (image.width * scale, image.height * scale)
        result = image.resize(new_size, Image.LANCZOS)
        return result, f"Lanczos 리사이즈 완료 ({scale}x)"
    
    def get_available_models(self) -> dict:
        """사용 가능한 모델 목록"""
        return {name: info["description"] for name, info in self.MODELS.items()}


# 전역 인스턴스
upscaler = Upscaler()
