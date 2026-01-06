"""애플리케이션 전역 상태 관리"""

import asyncio
import time


class AppState:
    """애플리케이션 전역 상태
    
    모델 인스턴스, 잠금, 활동 시간 등을 중앙에서 관리합니다.
    여러 라우터에서 이 상태에 접근합니다.
    """
    
    def __init__(self):
        # Z-Image 생성 모델 관련
        self.pipe = None
        self.current_model = None
        self.device = None
        self.model_lock = asyncio.Lock()
        self.last_activity_time = time.time()
        
        # Qwen-Image-Edit 편집 모델 관련
        self.edit_last_activity_time = time.time()
        self.edit_model_lock = asyncio.Lock()
    
    def update_activity(self):
        """마지막 활동 시간 업데이트 (생성 모델)"""
        self.last_activity_time = time.time()
    
    def update_edit_activity(self):
        """마지막 활동 시간 업데이트 (편집 모델)"""
        self.edit_last_activity_time = time.time()
    
    @property
    def is_model_loaded(self) -> bool:
        """생성 모델 로드 여부"""
        return self.pipe is not None
    
    def get_vram_info(self) -> str:
        """VRAM 사용량 정보"""
        from utils.gpu_monitor import gpu_monitor
        return gpu_monitor.get_vram_summary()


# 전역 싱글톤 인스턴스
app_state = AppState()
