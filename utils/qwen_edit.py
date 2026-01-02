"""Qwen-Image-Edit 파이프라인 관리

ovedrive/Qwen-Image-Edit-2511-4bit 모델을 사용한 이미지 편집
- 1~3장의 이미지 입력 지원
- 4bit NF4 양자화 (별도 양자화 옵션 불필요)
- true_cfg_scale 파라미터 지원
"""

import gc
import asyncio
from typing import Optional, Tuple, Callable, Any, List

import torch
from PIL import Image

from config.defaults import (
    QWEN_EDIT_MODEL,
    DEFAULT_QWEN_EDIT_SETTINGS,
    DEFAULT_GPU_SETTINGS,
)
from utils.gpu_monitor import gpu_monitor


class QwenEditManager:
    """Qwen-Image-Edit 모델 관리자"""
    
    def __init__(self):
        self.pipe = None
        self.current_model: Optional[str] = None
        self.device: Optional[str] = None
        self.cpu_offload_enabled: bool = False
        self._lock = asyncio.Lock()
        self._original_progress_bar = None
    
    @property
    def is_loaded(self) -> bool:
        """모델 로드 여부"""
        return self.pipe is not None
    
    def get_device(self, target_device: str = "auto") -> str:
        """
        사용할 디바이스 반환
        
        Args:
            target_device: 목표 디바이스 ("auto", "cuda:0", "cuda:1", "cpu", "mps")
        
        Returns:
            실제 사용할 디바이스
        """
        return gpu_monitor.resolve_device(target_device, prefer_empty=True)
    
    def _get_cuda_index_from_device(self, device: str) -> int:
        """'cuda', 'cuda:0' 형태에서 index 추출 (기본 0)"""
        if not device or not device.startswith("cuda"):
            return 0
        if ":" not in device:
            return 0
        try:
            return int(device.split(":", 1)[1])
        except Exception:
            return 0

    def _get_preferred_dtype(self, device: Optional[str]) -> torch.dtype:
        """
        GPU/플랫폼에 맞는 dtype 선택
        - Ampere(8.x)+: bf16 우선
        - 그 외 CUDA: fp16
        - CPU: fp32
        """
        if device and device.startswith("cuda") and torch.cuda.is_available():
            idx = self._get_cuda_index_from_device(device)
            try:
                major, _minor = torch.cuda.get_device_capability(idx)
            except Exception:
                major = 0
            return torch.bfloat16 if major >= 8 else torch.float16
        return torch.float32
    
    async def load_model(
        self,
        cpu_offload: bool = True,
        model_path: Optional[str] = None,
        target_device: str = "auto",
        progress_callback: Optional[Callable[[int, str, str], Any]] = None
    ) -> Tuple[bool, str]:
        """
        Qwen-Image-Edit 모델 로드
        
        Args:
            cpu_offload: CPU 오프로딩 사용 여부 (VRAM 절약)
            model_path: 커스텀 모델 경로
            target_device: 목표 디바이스 ("auto", "cuda:0", "cuda:1", "cpu", "mps")
            progress_callback: 진행상황 콜백 (percent, label, detail)
        
        Returns:
            (success, message)
        """
        async with self._lock:
            if self.pipe is not None:
                return False, "모델이 이미 로드되어 있습니다. 먼저 언로드하세요."
            
            try:
                self.device = self.get_device(target_device)
                preferred_dtype = self._get_preferred_dtype(self.device)
                
                # 진행 상황 콜백
                def report_progress(percent: int, label: str, detail: str = ""):
                    if progress_callback:
                        if asyncio.iscoroutinefunction(progress_callback):
                            asyncio.create_task(progress_callback(percent, label, detail))
                        else:
                            asyncio.create_task(
                                asyncio.to_thread(progress_callback, percent, label, detail)
                            )
                
                report_progress(5, "🔧 Qwen-Image-Edit 모델 초기화 중...", f"디바이스: {self.device}, 양자화: NF4 (4bit)")
                
                # diffusers에서 파이프라인 임포트
                from diffusers import QwenImageEditPlusPipeline
                
                checkpoint_dir = model_path if model_path else QWEN_EDIT_MODEL
                
                report_progress(10, "📥 모델 다운로드 확인 중...", f"저장소: {checkpoint_dir}")
                
                # 파이프라인 로드 (4bit 양자화 모델)
                report_progress(30, "🔄 파이프라인 로딩 중...", "대용량 모델 로드 중 (시간이 걸릴 수 있습니다)")
                
                self.pipe = await asyncio.to_thread(
                    QwenImageEditPlusPipeline.from_pretrained,
                    checkpoint_dir,
                    torch_dtype=preferred_dtype,
                )
                
                # 디바이스 설정
                report_progress(80, f"🚀 {self.device.upper()}로 모델 전송 중...", "")
                
                if cpu_offload:
                    await asyncio.to_thread(self.pipe.enable_model_cpu_offload)
                    report_progress(95, "⚙️ CPU 오프로딩 설정됨", "VRAM 부족 시 RAM 사용")
                    self.cpu_offload_enabled = True
                else:
                    await asyncio.to_thread(self.pipe.to, self.device)
                    self.cpu_offload_enabled = False
                
                # progress bar 설정
                self.pipe.set_progress_bar_config(disable=None)
                
                self.current_model = QWEN_EDIT_MODEL
                
                # 원본 progress_bar 메서드 저장 (후킹 복원용)
                self._original_progress_bar = self.pipe.progress_bar.__func__
                
                # GPU 모니터에 모델 등록
                gpu_monitor.register_model("Qwen-Image-Edit", self.device)
                
                report_progress(100, "✅ Qwen-Image-Edit 모델 로드 완료!", self._get_vram_info())
                
                return True, f"모델 로드 완료: {checkpoint_dir} (NF4 4bit)"
                
            except ImportError as e:
                error_msg = str(e)
                if "QwenImageEditPlusPipeline" in error_msg:
                    return False, "diffusers 최신 버전이 필요합니다. 'pip install git+https://github.com/huggingface/diffusers'를 실행하세요."
                return False, f"모델 로드 실패: {str(e)}"
            except Exception as e:
                self._cleanup()
                return False, f"모델 로드 실패: {str(e)}"
    
    async def unload_model(self) -> Tuple[bool, str]:
        """모델 언로드"""
        async with self._lock:
            if self.pipe is None:
                return True, "로드된 모델이 없습니다."
            
            try:
                self._cleanup()
                return True, "모델 언로드 완료"
            except Exception as e:
                return False, f"모델 언로드 실패: {str(e)}"
    
    def _cleanup(self):
        """모델 메모리 정리"""
        # GPU 모니터에서 모델 등록 해제
        gpu_monitor.unregister_model("Qwen-Image-Edit")
        
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        
        self.current_model = None
        self.cpu_offload_enabled = False
        self._original_progress_bar = None
        
        # GPU 캐시 정리
        gpu_monitor.clear_cache(self.device)
        gc.collect()
    
    def _get_vram_info(self) -> str:
        """VRAM 사용량 정보"""
        return gpu_monitor.get_vram_summary()
    
    def _hook_progress_bar(self, step_callback):
        """파이프라인의 progress_bar를 후킹하여 스텝별 콜백 호출"""
        pipe = self.pipe
        
        if self._original_progress_bar is None:
            print("[경고] 원본 progress_bar가 저장되지 않음, 현재 메서드 사용")
            original_progress_bar = pipe.progress_bar.__func__
        else:
            original_progress_bar = self._original_progress_bar
        
        def hooked_progress_bar(self_pipe, *args, **kwargs):
            pbar = original_progress_bar(self_pipe, *args, **kwargs)
            
            original_update = pbar.update
            
            def hooked_update(n=1):
                result = original_update(n)
                if step_callback and pbar.total:
                    step_callback(pbar.n, pbar.total)
                return result
            
            pbar.update = hooked_update
            return pbar
        
        import types
        pipe.progress_bar = types.MethodType(hooked_progress_bar, pipe)
    
    def _restore_progress_bar(self):
        """원래 progress_bar 복원"""
        import types
        if self.pipe and self._original_progress_bar:
            self.pipe.progress_bar = types.MethodType(self._original_progress_bar, self.pipe)
    
    async def edit_image(
        self,
        images: List[Image.Image],
        prompt: str,
        negative_prompt: str = " ",
        num_inference_steps: int = 20,
        true_cfg_scale: float = 4.0,
        guidance_scale: float = 1.0,
        seed: int = -1,
        num_images: int = 1,
        progress_callback: Optional[Callable[[int, int, int, int], Any]] = None,
        status_callback: Optional[Callable[[str], Any]] = None
    ) -> Tuple[bool, list, str]:
        """
        이미지 편집 실행
        
        Args:
            images: 편집할 원본 이미지 리스트 (1~3장)
            prompt: 편집 프롬프트
            negative_prompt: 네거티브 프롬프트
            num_inference_steps: 추론 스텝 수 (기본 20)
            true_cfg_scale: True CFG 스케일 (기본 4.0, 프롬프트 충실도)
            guidance_scale: 가이던스 스케일 (기본 1.0)
            seed: 시드 (-1이면 랜덤)
            num_images: 생성할 이미지 수
            progress_callback: 진행상황 콜백 (current_image, total_images, current_step, total_steps)
            status_callback: 상태 메시지 콜백 (message)
        
        Returns:
            (success, images, message)
        """
        if self.pipe is None:
            return False, [], "모델이 로드되지 않았습니다."
        
        if not images or len(images) == 0:
            return False, [], "최소 1장의 이미지가 필요합니다."
        
        if len(images) > 3:
            return False, [], "최대 3장의 이미지만 입력할 수 있습니다."
        
        try:
            import random
            
            # 편집 시작 전 GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            
            # RGB로 변환
            processed_images = []
            for img in images:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                processed_images.append(img)
            
            # 시드 설정
            if seed == -1:
                seed = random.randint(0, 2147483647)
            
            # 메인 이벤트 루프 캡처
            main_loop = asyncio.get_running_loop()
            
            results = []
            for i in range(num_images):
                current_seed = seed + i
                generator = torch.Generator("cpu").manual_seed(current_seed)
                
                # 스텝별 콜백 함수 생성
                current_image_idx = i
                total_images = num_images
                
                def create_step_callback(img_idx, total_imgs):
                    def step_callback(current_step, total_steps):
                        if progress_callback:
                            asyncio.run_coroutine_threadsafe(
                                progress_callback(img_idx + 1, total_imgs, current_step, total_steps),
                                main_loop
                            )
                    return step_callback
                
                step_cb = create_step_callback(current_image_idx, total_images)
                
                # progress_bar 후킹
                self._hook_progress_bar(step_cb)
                
                try:
                    def run_edit():
                        return self.pipe(
                            image=processed_images,
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            num_inference_steps=num_inference_steps,
                            true_cfg_scale=true_cfg_scale,
                            guidance_scale=guidance_scale,
                            generator=generator,
                            num_images_per_prompt=1,
                        ).images[0]
                    
                    result_image = await asyncio.to_thread(run_edit)
                finally:
                    self._restore_progress_bar()
                
                results.append({
                    "image": result_image,
                    "seed": current_seed
                })
            
            # 편집 완료 후 GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            
            return True, results, f"편집 완료 (시드: {seed})"
            
        except Exception as e:
            self._restore_progress_bar()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            return False, [], f"편집 실패: {str(e)}"


# 전역 인스턴스
qwen_edit_manager = QwenEditManager()
