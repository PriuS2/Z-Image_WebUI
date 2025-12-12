"""LongCat-Image-Edit 파이프라인 관리"""

import gc
import asyncio
from typing import Optional, Tuple, Callable, Any

import torch
from PIL import Image

from config.defaults import (
    LONGCAT_EDIT_MODEL,
    EDIT_QUANTIZATION_OPTIONS,
    DEFAULT_EDIT_SETTINGS,
)


class LongCatEditManager:
    """LongCat-Image-Edit 모델 관리자"""
    
    def __init__(self):
        self.pipe = None
        self.transformer = None
        self.text_processor = None
        self.current_model: Optional[str] = None
        self.device: Optional[str] = None
        self._lock = asyncio.Lock()
    
    @property
    def is_loaded(self) -> bool:
        """모델 로드 여부"""
        return self.pipe is not None
    
    def get_device(self) -> str:
        """사용 가능한 디바이스 반환"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    async def load_model(
        self,
        quantization: str = "BF16 (기본, 최고품질)",
        cpu_offload: bool = True,
        model_path: Optional[str] = None,
        progress_callback: Optional[Callable[[int, str, str], Any]] = None
    ) -> Tuple[bool, str]:
        """
        LongCat-Image-Edit 모델 로드
        
        Args:
            quantization: 양자화 옵션
            cpu_offload: CPU 오프로딩 사용 여부 (VRAM 절약)
            model_path: 커스텀 모델 경로
            progress_callback: 진행상황 콜백 (percent, label, detail)
        
        Returns:
            (success, message)
        """
        async with self._lock:
            if self.pipe is not None:
                return False, "모델이 이미 로드되어 있습니다. 먼저 언로드하세요."
            
            try:
                self.device = self.get_device()
                quant_info = EDIT_QUANTIZATION_OPTIONS.get(quantization)
                
                if not quant_info:
                    return False, f"지원하지 않는 양자화: {quantization}"
                
                repo_id = quant_info["repo"]
                
                # 진행 상황 콜백
                def report_progress(percent: int, label: str, detail: str = ""):
                    if progress_callback:
                        # async 함수와 sync 함수 모두 지원
                        if asyncio.iscoroutinefunction(progress_callback):
                            asyncio.create_task(progress_callback(percent, label, detail))
                        else:
                            asyncio.create_task(
                                asyncio.to_thread(progress_callback, percent, label, detail)
                            )
                
                report_progress(5, "🔧 LongCat-Image-Edit 모델 초기화 중...", f"디바이스: {self.device}")
                
                # LongCat-Image 패키지에서 임포트
                from transformers import AutoProcessor
                from longcat_image.models import LongCatImageTransformer2DModel
                from longcat_image.pipelines import LongCatImageEditPipeline
                
                checkpoint_dir = model_path if model_path else repo_id
                
                # BF16 모델 로드
                report_progress(10, "📥 모델 다운로드 확인 중...", f"저장소: {checkpoint_dir}")
                
                # Text Processor 로드
                report_progress(20, "🔄 Text Processor 로딩 중...", "")
                self.text_processor = await asyncio.to_thread(
                    AutoProcessor.from_pretrained,
                    checkpoint_dir,
                    subfolder="tokenizer"
                )
                
                # Transformer 로드
                report_progress(40, "🔄 Transformer 로딩 중...", "대용량 모델 로드 중 (시간이 걸릴 수 있습니다)")
                self.transformer = await asyncio.to_thread(
                    LongCatImageTransformer2DModel.from_pretrained,
                    checkpoint_dir,
                    subfolder="transformer",
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True
                )
                
                # 파이프라인 구성
                report_progress(70, "🔗 파이프라인 구성 중...", "")
                self.pipe = await asyncio.to_thread(
                    LongCatImageEditPipeline.from_pretrained,
                    checkpoint_dir,
                    transformer=self.transformer,
                    text_processor=self.text_processor,
                    torch_dtype=torch.bfloat16
                )
                
                # 디바이스 설정
                report_progress(85, f"🚀 {self.device.upper()}로 모델 전송 중...", "")
                
                if cpu_offload:
                    await asyncio.to_thread(self.pipe.enable_model_cpu_offload)
                    report_progress(95, "⚙️ CPU 오프로딩 설정됨", "VRAM 부족 시 RAM 사용")
                else:
                    await asyncio.to_thread(self.pipe.to, self.device, torch.bfloat16)
                
                self.current_model = quantization
                
                report_progress(100, "✅ LongCat-Image-Edit 모델 로드 완료!", self._get_vram_info())
                
                return True, f"모델 로드 완료: {checkpoint_dir}"
                
            except ImportError as e:
                return False, f"LongCat-Image 패키지가 설치되지 않았습니다. 'pip install -e ./LongCat-Image'를 실행하세요. 오류: {e}"
            except Exception as e:
                # 실패 시 정리
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
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        
        if self.transformer is not None:
            del self.transformer
            self.transformer = None
        
        if self.text_processor is not None:
            del self.text_processor
            self.text_processor = None
        
        self.current_model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        gc.collect()
    
    def _get_vram_info(self) -> str:
        """VRAM 사용량 정보"""
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1024**3
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return f"VRAM: {vram_used:.1f}GB / {vram_total:.1f}GB"
        return "N/A"
    
    def _hook_progress_bar(self, step_callback):
        """파이프라인의 progress_bar를 후킹하여 스텝별 콜백 호출"""
        from tqdm import tqdm
        import types
        
        pipe = self.pipe
        original_progress_bar = pipe.progress_bar
        
        def hooked_progress_bar(*args, **kwargs):
            # 원래 progress_bar 호출
            pbar = original_progress_bar(*args, **kwargs)
            
            # tqdm의 update 메서드를 후킹
            original_update = pbar.update
            
            def hooked_update(n=1):
                result = original_update(n)
                # 스텝 콜백 호출
                if step_callback and pbar.total:
                    step_callback(pbar.n, pbar.total)
                return result
            
            pbar.update = hooked_update
            return pbar
        
        # 메서드 바인딩
        pipe.progress_bar = types.MethodType(
            lambda self, *args, **kwargs: hooked_progress_bar(*args, **kwargs),
            pipe
        )
        
        return original_progress_bar
    
    def _restore_progress_bar(self, original_progress_bar):
        """원래 progress_bar 복원"""
        import types
        if self.pipe and original_progress_bar:
            self.pipe.progress_bar = types.MethodType(original_progress_bar, self.pipe)
    
    async def edit_image(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 4.5,
        seed: int = -1,
        num_images: int = 1,
        reference_image: Optional[Image.Image] = None,
        progress_callback: Optional[Callable[[int, int, int, int], Any]] = None
    ) -> Tuple[bool, list, str]:
        """
        이미지 편집 실행
        
        Args:
            image: 편집할 원본 이미지
            prompt: 편집 프롬프트
            negative_prompt: 네거티브 프롬프트
            num_inference_steps: 추론 스텝 수
            guidance_scale: 가이던스 스케일
            seed: 시드 (-1이면 랜덤)
            num_images: 생성할 이미지 수
            reference_image: 참조 이미지 (스타일 참조용)
            progress_callback: 진행상황 콜백 (current_image, total_images, current_step, total_steps)
        
        Returns:
            (success, images, message)
        """
        if self.pipe is None:
            return False, [], "모델이 로드되지 않았습니다."
        
        try:
            import random
            
            # RGB로 변환
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # 시드 설정
            if seed == -1:
                seed = random.randint(0, 2147483647)
            
            generator = torch.Generator("cpu").manual_seed(seed)
            
            results = []
            for i in range(num_images):
                current_seed = seed + i
                if i > 0:
                    generator = torch.Generator("cpu").manual_seed(current_seed)
                
                # 스텝 콜백을 위한 상태 저장
                current_image_idx = i
                total_images = num_images
                
                # 스텝별 콜백 함수
                def step_callback(current_step, total_steps):
                    if progress_callback:
                        # sync 콜백 호출 (별도 스레드에서 실행되므로)
                        try:
                            # 이벤트 루프가 있으면 asyncio로 실행
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                asyncio.run_coroutine_threadsafe(
                                    progress_callback(current_image_idx + 1, total_images, current_step, total_steps),
                                    loop
                                )
                        except RuntimeError:
                            pass  # 이벤트 루프가 없으면 무시
                
                # progress_bar 후킹
                original_progress_bar = self._hook_progress_bar(step_callback)
                
                try:
                    # 편집 실행
                    def run_edit():
                        return self.pipe(
                            image,
                            prompt,
                            negative_prompt=negative_prompt,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps,
                            num_images_per_prompt=1,
                            generator=generator
                        ).images[0]
                    
                    result_image = await asyncio.to_thread(run_edit)
                finally:
                    # progress_bar 복원
                    self._restore_progress_bar(original_progress_bar)
                
                results.append({
                    "image": result_image,
                    "seed": current_seed
                })
            
            return True, results, f"편집 완료 (시드: {seed})"
            
        except Exception as e:
            return False, [], f"편집 실패: {str(e)}"


# 전역 인스턴스
longcat_edit_manager = LongCatEditManager()

