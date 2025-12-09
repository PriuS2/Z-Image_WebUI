"""ControlNet 전처리기 모듈

각 컨트롤 타입에 맞는 이미지 전처리를 수행합니다:
- Canny: 엣지 검출
- Depth: 깊이 맵 추출
- Pose: 포즈 스켈레톤 추출
- HED: 홀리스틱 엣지 검출
- MLSD: 직선 세그먼트 검출
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Union
from io import BytesIO
import base64

# 전처리기 가용성 플래그
CONTROLNET_AUX_AVAILABLE = False
MEDIAPIPE_AVAILABLE = False

try:
    from controlnet_aux import (
        CannyDetector,
        HEDdetector,
        MLSDdetector,
        MidasDetector,
        OpenposeDetector,
    )
    CONTROLNET_AUX_AVAILABLE = True
except ImportError:
    print("⚠️ controlnet-aux가 설치되지 않았습니다. pip install controlnet-aux")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("⚠️ mediapipe가 설치되지 않았습니다. pip install mediapipe")


# 컨트롤 타입 정의
CONTROL_TYPES = {
    "canny": {
        "name": "Canny Edge",
        "description": "윤곽선 추출 (날카로운 엣지)",
        "icon": "ri-pencil-ruler-2-line",
    },
    "depth": {
        "name": "Depth Map",
        "description": "깊이 맵 추출 (원근감)",
        "icon": "ri-3d-cude-sphere-line",
    },
    "pose": {
        "name": "Pose",
        "description": "인체 포즈 스켈레톤 추출",
        "icon": "ri-body-scan-line",
    },
    "hed": {
        "name": "HED Edge",
        "description": "부드러운 윤곽선 추출",
        "icon": "ri-brush-line",
    },
    "mlsd": {
        "name": "MLSD Lines",
        "description": "직선 구조 추출 (건축물)",
        "icon": "ri-layout-grid-line",
    },
}


class ControlNetPreprocessor:
    """ControlNet 전처리기 클래스"""
    
    def __init__(self):
        self._canny_detector = None
        self._hed_detector = None
        self._mlsd_detector = None
        self._midas_detector = None
        self._openpose_detector = None
        self._mp_pose = None
    
    @property
    def is_available(self) -> bool:
        """전처리기 사용 가능 여부"""
        return CONTROLNET_AUX_AVAILABLE
    
    def get_canny_detector(self):
        """Canny 검출기 (지연 로딩)"""
        if self._canny_detector is None and CONTROLNET_AUX_AVAILABLE:
            self._canny_detector = CannyDetector()
        return self._canny_detector
    
    def get_hed_detector(self):
        """HED 검출기 (지연 로딩)"""
        if self._hed_detector is None and CONTROLNET_AUX_AVAILABLE:
            self._hed_detector = HEDdetector.from_pretrained("lllyasviel/Annotators")
        return self._hed_detector
    
    def get_mlsd_detector(self):
        """MLSD 검출기 (지연 로딩)"""
        if self._mlsd_detector is None and CONTROLNET_AUX_AVAILABLE:
            self._mlsd_detector = MLSDdetector.from_pretrained("lllyasviel/Annotators")
        return self._mlsd_detector
    
    def get_midas_detector(self):
        """MiDaS 깊이 검출기 (지연 로딩)"""
        if self._midas_detector is None and CONTROLNET_AUX_AVAILABLE:
            self._midas_detector = MidasDetector.from_pretrained("lllyasviel/Annotators")
        return self._midas_detector
    
    def get_openpose_detector(self):
        """OpenPose 검출기 (지연 로딩)"""
        if self._openpose_detector is None and CONTROLNET_AUX_AVAILABLE:
            self._openpose_detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
        return self._openpose_detector
    
    def process_canny(
        self,
        image: Image.Image,
        low_threshold: int = 100,
        high_threshold: int = 200,
    ) -> Image.Image:
        """
        Canny 엣지 검출
        
        Args:
            image: 입력 이미지
            low_threshold: 낮은 임계값 (기본: 100)
            high_threshold: 높은 임계값 (기본: 200)
        
        Returns:
            Canny 엣지 이미지
        """
        if CONTROLNET_AUX_AVAILABLE:
            detector = self.get_canny_detector()
            return detector(image, low_threshold, high_threshold)
        else:
            # OpenCV 폴백
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            edges = cv2.Canny(gray, low_threshold, high_threshold)
            return Image.fromarray(edges)
    
    def process_depth(self, image: Image.Image) -> Image.Image:
        """
        깊이 맵 추출 (MiDaS)
        
        Args:
            image: 입력 이미지
        
        Returns:
            깊이 맵 이미지
        """
        if not CONTROLNET_AUX_AVAILABLE:
            raise RuntimeError("controlnet-aux가 필요합니다: pip install controlnet-aux")
        
        detector = self.get_midas_detector()
        return detector(image)
    
    def process_pose(
        self,
        image: Image.Image,
        include_hand: bool = False,
        include_face: bool = False,
    ) -> Image.Image:
        """
        포즈 스켈레톤 추출
        
        Args:
            image: 입력 이미지
            include_hand: 손 포즈 포함 여부
            include_face: 얼굴 포즈 포함 여부
        
        Returns:
            포즈 스켈레톤 이미지
        """
        if not CONTROLNET_AUX_AVAILABLE:
            raise RuntimeError("controlnet-aux가 필요합니다: pip install controlnet-aux")
        
        detector = self.get_openpose_detector()
        return detector(image, include_hand=include_hand, include_face=include_face)
    
    def process_hed(self, image: Image.Image) -> Image.Image:
        """
        HED (Holistically-nested Edge Detection)
        
        Args:
            image: 입력 이미지
        
        Returns:
            HED 엣지 이미지
        """
        if not CONTROLNET_AUX_AVAILABLE:
            raise RuntimeError("controlnet-aux가 필요합니다: pip install controlnet-aux")
        
        detector = self.get_hed_detector()
        return detector(image)
    
    def process_mlsd(
        self,
        image: Image.Image,
        thr_v: float = 0.1,
        thr_d: float = 0.1,
    ) -> Image.Image:
        """
        MLSD (Mobile Line Segment Detection)
        
        Args:
            image: 입력 이미지
            thr_v: 밸류 임계값
            thr_d: 거리 임계값
        
        Returns:
            MLSD 직선 이미지
        """
        if not CONTROLNET_AUX_AVAILABLE:
            raise RuntimeError("controlnet-aux가 필요합니다: pip install controlnet-aux")
        
        detector = self.get_mlsd_detector()
        return detector(image, thr_v=thr_v, thr_d=thr_d)
    
    def preprocess(
        self,
        image: Image.Image,
        control_type: str,
        **kwargs,
    ) -> Image.Image:
        """
        컨트롤 타입에 따라 전처리 수행
        
        Args:
            image: 입력 이미지
            control_type: 컨트롤 타입 (canny, depth, pose, hed, mlsd)
            **kwargs: 각 전처리기별 추가 파라미터
        
        Returns:
            전처리된 이미지
        """
        processors = {
            "canny": self.process_canny,
            "depth": self.process_depth,
            "pose": self.process_pose,
            "hed": self.process_hed,
            "mlsd": self.process_mlsd,
        }
        
        if control_type not in processors:
            raise ValueError(f"지원하지 않는 컨트롤 타입: {control_type}")
        
        return processors[control_type](image, **kwargs)
    
    def resize_for_condition(
        self,
        image: Image.Image,
        target_width: int,
        target_height: int,
    ) -> Image.Image:
        """
        컨트롤 이미지를 생성 해상도에 맞게 리사이즈
        
        Args:
            image: 입력 이미지
            target_width: 목표 너비
            target_height: 목표 높이
        
        Returns:
            리사이즈된 이미지
        """
        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


# 전역 인스턴스
preprocessor = ControlNetPreprocessor()


def image_to_base64(image: Image.Image) -> str:
    """PIL 이미지를 base64로 변환"""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def base64_to_image(base64_str: str) -> Image.Image:
    """base64 문자열을 PIL 이미지로 변환"""
    # data:image/png;base64, 접두사 제거
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data)).convert("RGB")

