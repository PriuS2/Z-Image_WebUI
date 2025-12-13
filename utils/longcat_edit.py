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
    DEFAULT_GPU_SETTINGS,
)
from utils.llm_client import llm_client
from utils.gpu_monitor import gpu_monitor


# 참조 이미지 분석용 프롬프트 템플릿 (편집 프롬프트 기반으로 필요한 요소만 추출)
REFERENCE_IMAGE_ANALYSIS_TEMPLATE = """You are an expert at analyzing images for AI image editing tasks.

The user wants to edit an image with this instruction: "{edit_prompt}"

Your task: Look at the reference image and extract ONLY the specific elements mentioned or implied in the edit instruction.

Rules:
1. Focus ONLY on elements relevant to the edit instruction
2. If the instruction mentions "flower pot" or "plant", describe ONLY the pot/plant from the reference image
3. If the instruction mentions "style" or "atmosphere", describe ONLY the artistic style
4. If the instruction mentions a specific object, describe ONLY that object
5. Do NOT describe the entire image - be selective and focused
6. Keep description brief (max 50 words) - just the essential visual details

Examples:
- Edit instruction: "Make them hold this flower pot" → Describe only the pot (color, material, plant type)
- Edit instruction: "Apply this painting style" → Describe only the art style (brushwork, colors, technique)
- Edit instruction: "Add this hat to the person" → Describe only the hat (shape, color, material)

Output ONLY a brief, focused description of the relevant element(s). No explanation or preamble.
Output in English."""


class LongCatEditManager:
    """LongCat-Image-Edit 모델 관리자"""
    
    def __init__(self):
        self.pipe = None
        self.transformer = None
        self.text_processor = None
        self.text_encoder = None  # 양자화된 text_encoder 별도 관리
        self.current_model: Optional[str] = None
        self.current_quantization: Optional[str] = None  # 현재 양자화 타입
        self.device: Optional[str] = None
        self.cpu_offload_enabled: bool = False  # CPU 오프로딩 활성화 여부
        self._lock = asyncio.Lock()
        self._original_progress_bar = None  # 원본 progress_bar 메서드 저장
    
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
    
    def _check_bitsandbytes_available(self) -> Tuple[bool, str]:
        """bitsandbytes 라이브러리 사용 가능 여부 확인"""
        try:
            import bitsandbytes as bnb
            return True, ""
        except ImportError:
            return False, "bitsandbytes 라이브러리가 설치되지 않았습니다. 'pip install bitsandbytes'를 실행하세요."
    
    def _get_quantization_config(self, quantization_type: str):
        """양자화 설정 반환"""
        if quantization_type == "int8":
            from transformers import BitsAndBytesConfig
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
        elif quantization_type == "int4":
            from transformers import BitsAndBytesConfig
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        return None
    
    async def load_model(
        self,
        quantization: str = "BF16 (기본, 최고품질)",
        cpu_offload: bool = True,
        model_path: Optional[str] = None,
        target_device: str = "auto",
        progress_callback: Optional[Callable[[int, str, str], Any]] = None
    ) -> Tuple[bool, str]:
        """
        LongCat-Image-Edit 모델 로드
        
        Args:
            quantization: 양자화 옵션 (BF16, INT8, INT4)
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
                quant_info = EDIT_QUANTIZATION_OPTIONS.get(quantization)
                
                if not quant_info:
                    return False, f"지원하지 않는 양자화: {quantization}"
                
                repo_id = quant_info["repo"]
                quantization_type = quant_info.get("quantization")  # None, "int8", "int4"
                
                # 양자화 사용 시 bitsandbytes 확인
                if quantization_type in ("int8", "int4"):
                    bnb_available, bnb_error = self._check_bitsandbytes_available()
                    if not bnb_available:
                        return False, bnb_error
                
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
                
                quant_label = quantization_type.upper() if quantization_type else "BF16"
                report_progress(5, f"🔧 LongCat-Image-Edit 모델 초기화 중...", f"디바이스: {self.device}, 양자화: {quant_label}")
                
                # LongCat-Image 패키지에서 임포트
                from transformers import AutoProcessor, AutoModel
                from longcat_image.models import LongCatImageTransformer2DModel
                from longcat_image.pipelines import LongCatImageEditPipeline
                
                checkpoint_dir = model_path if model_path else repo_id
                
                report_progress(10, "📥 모델 다운로드 확인 중...", f"저장소: {checkpoint_dir}")
                
                # Text Processor 로드
                report_progress(15, "🔄 Text Processor 로딩 중...", "")
                self.text_processor = await asyncio.to_thread(
                    AutoProcessor.from_pretrained,
                    checkpoint_dir,
                    subfolder="tokenizer"
                )
                
                # Text Encoder 로드 (양자화 적용)
                if quantization_type in ("int8", "int4"):
                    report_progress(25, f"🔄 Text Encoder 로딩 중 ({quant_label} 양자화)...", "VRAM 절약 모드")
                    
                    quant_config = self._get_quantization_config(quantization_type)
                    
                    # 양자화된 text_encoder 로드
                    def load_quantized_encoder():
                        from transformers import Qwen2VLForConditionalGeneration
                        return Qwen2VLForConditionalGeneration.from_pretrained(
                            checkpoint_dir,
                            subfolder="text_encoder",
                            quantization_config=quant_config,
                            torch_dtype=torch.bfloat16,
                            device_map="auto" if not cpu_offload else None,
                        )
                    
                    self.text_encoder = await asyncio.to_thread(load_quantized_encoder)
                else:
                    self.text_encoder = None
                
                # Transformer 로드
                report_progress(45, "🔄 Transformer 로딩 중...", "대용량 모델 로드 중 (시간이 걸릴 수 있습니다)")
                self.transformer = await asyncio.to_thread(
                    LongCatImageTransformer2DModel.from_pretrained,
                    checkpoint_dir,
                    subfolder="transformer",
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True
                )
                
                # 파이프라인 구성
                report_progress(70, "🔗 파이프라인 구성 중...", "")
                
                pipeline_kwargs = {
                    "transformer": self.transformer,
                    "text_processor": self.text_processor,
                    "torch_dtype": torch.bfloat16,
                }
                
                # 양자화된 text_encoder가 있으면 사용
                if self.text_encoder is not None:
                    pipeline_kwargs["text_encoder"] = self.text_encoder
                
                self.pipe = await asyncio.to_thread(
                    LongCatImageEditPipeline.from_pretrained,
                    checkpoint_dir,
                    **pipeline_kwargs
                )
                
                # VAE 메모리 최적화 활성화
                report_progress(80, "🔧 메모리 최적화 설정 중...", "VAE slicing/tiling 활성화")
                await asyncio.to_thread(self.pipe.enable_vae_slicing)
                await asyncio.to_thread(self.pipe.enable_vae_tiling)
                
                # 디바이스 설정
                report_progress(85, f"🚀 {self.device.upper()}로 모델 전송 중...", "")
                
                if cpu_offload:
                    # 양자화된 모델은 device_map이 이미 설정되어 있을 수 있음
                    if self.text_encoder is None:
                        await asyncio.to_thread(self.pipe.enable_model_cpu_offload)
                    else:
                        # 양자화 + CPU 오프로딩: transformer와 VAE만 오프로딩
                        await asyncio.to_thread(self.pipe.enable_model_cpu_offload)
                    report_progress(95, "⚙️ CPU 오프로딩 설정됨", "VRAM 부족 시 RAM 사용")
                    self.cpu_offload_enabled = True
                else:
                    # 양자화된 모델은 이미 device_map으로 배치됨
                    if self.text_encoder is None:
                        await asyncio.to_thread(self.pipe.to, self.device, torch.bfloat16)
                    self.cpu_offload_enabled = False
                
                self.current_model = quantization
                self.current_quantization = quantization_type
                
                # GPU 모니터에 모델 등록
                gpu_monitor.register_model("LongCat-Image-Edit", self.device)
                
                # 원본 progress_bar 메서드 저장 (후킹 복원용)
                self._original_progress_bar = self.pipe.progress_bar.__func__
                
                report_progress(100, "✅ LongCat-Image-Edit 모델 로드 완료!", self._get_vram_info())
                
                return True, f"모델 로드 완료: {checkpoint_dir} ({quant_label})"
                
            except ImportError as e:
                error_msg = str(e)
                if "bitsandbytes" in error_msg:
                    return False, "bitsandbytes 라이브러리가 설치되지 않았습니다. 'pip install bitsandbytes'를 실행하세요."
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
        # GPU 모니터에서 모델 등록 해제
        gpu_monitor.unregister_model("LongCat-Image-Edit")
        
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        
        if self.transformer is not None:
            del self.transformer
            self.transformer = None
        
        if self.text_processor is not None:
            del self.text_processor
            self.text_processor = None
        
        if self.text_encoder is not None:
            del self.text_encoder
            self.text_encoder = None
        
        self.current_model = None
        self.current_quantization = None
        self.cpu_offload_enabled = False
        self._original_progress_bar = None  # 원본 progress_bar 참조도 정리
        
        # GPU 캐시 정리
        gpu_monitor.clear_cache(self.device)
        gc.collect()
    
    def _get_vram_info(self) -> str:
        """VRAM 사용량 정보"""
        return gpu_monitor.get_vram_summary()
    
    def _hook_progress_bar(self, step_callback):
        """파이프라인의 progress_bar를 후킹하여 스텝별 콜백 호출"""
        pipe = self.pipe
        
        # 로드 시 저장된 원본 progress_bar 사용
        if self._original_progress_bar is None:
            print("[경고] 원본 progress_bar가 저장되지 않음, 현재 메서드 사용")
            original_progress_bar = pipe.progress_bar.__func__
        else:
            original_progress_bar = self._original_progress_bar
        
        def hooked_progress_bar(self_pipe, *args, **kwargs):
            # 원래 progress_bar 호출
            pbar = original_progress_bar(self_pipe, *args, **kwargs)
            
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
        import types
        pipe.progress_bar = types.MethodType(hooked_progress_bar, pipe)
    
    def _restore_progress_bar(self):
        """원래 progress_bar 복원"""
        import types
        if self.pipe and self._original_progress_bar:
            self.pipe.progress_bar = types.MethodType(self._original_progress_bar, self.pipe)
    
    async def analyze_reference_image(
        self,
        reference_image: Image.Image,
        original_prompt: str
    ) -> Tuple[str, bool]:
        """
        참조 이미지를 Vision LLM으로 분석하여 프롬프트에 통합
        
        Args:
            reference_image: 참조 이미지
            original_prompt: 원본 프롬프트
        
        Returns:
            (결합된 프롬프트, 성공 여부)
        """
        if not llm_client.is_available:
            print("LLM 클라이언트를 사용할 수 없습니다. 원본 프롬프트 사용.")
            return original_prompt, False
        
        try:
            # 편집 프롬프트 기반으로 필요한 요소만 추출하는 프롬프트 생성
            analysis_prompt = REFERENCE_IMAGE_ANALYSIS_TEMPLATE.format(
                edit_prompt=original_prompt
            )
            
            # 참조 이미지 분석 (편집 프롬프트와 관련된 요소만)
            analysis = await asyncio.to_thread(
                llm_client.analyze_image,
                reference_image,
                analysis_prompt,
                temperature=0.3,  # 더 일관된 결과를 위해 낮춤
                max_tokens=100   # 간결한 설명만 필요
            )
            
            if not analysis:
                print("참조 이미지 분석 실패. 원본 프롬프트 사용.")
                return original_prompt, False
            
            # 프롬프트 결합 (원본 지시 + 참조 요소 설명)
            # 원본 프롬프트를 먼저 두고, 참조 요소는 보조 정보로 추가
            combined_prompt = f"{original_prompt}. Reference element: {analysis}"
            
            print(f"[편집 프롬프트] {original_prompt}")
            print(f"[추출된 참조 요소] {analysis}")
            print(f"[최종 프롬프트] {combined_prompt}")
            
            return combined_prompt, True
            
        except Exception as e:
            print(f"참조 이미지 분석 오류: {e}")
            return original_prompt, False
    
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
        progress_callback: Optional[Callable[[int, int, int, int], Any]] = None,
        status_callback: Optional[Callable[[str], Any]] = None
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
            reference_image: 참조 이미지 (스타일 참조용, Vision LLM으로 분석)
            progress_callback: 진행상황 콜백 (current_image, total_images, current_step, total_steps)
            status_callback: 상태 메시지 콜백 (message)
        
        Returns:
            (success, images, message)
        """
        if self.pipe is None:
            return False, [], "모델이 로드되지 않았습니다."
        
        try:
            import random
            
            # 편집 시작 전 GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            
            # RGB로 변환
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # 참조 이미지 분석 및 프롬프트 결합
            final_prompt = prompt
            if reference_image is not None:
                if status_callback:
                    await status_callback("🔍 참조 이미지 분석 중...")
                
                combined_prompt, success = await self.analyze_reference_image(
                    reference_image, prompt
                )
                
                if success:
                    final_prompt = combined_prompt
                    if status_callback:
                        await status_callback("✅ 참조 이미지 스타일 적용됨")
                        await status_callback(f"📝 최종 프롬프트: {final_prompt}")
                else:
                    if status_callback:
                        await status_callback("⚠️ 참조 이미지 분석 실패, 원본 프롬프트 사용")
            
            # 시드 설정
            if seed == -1:
                seed = random.randint(0, 2147483647)
            
            generator = torch.Generator("cpu").manual_seed(seed)
            
            # 메인 이벤트 루프 캡처 (별도 스레드에서 사용하기 위해)
            main_loop = asyncio.get_running_loop()
            
            results = []
            for i in range(num_images):
                current_seed = seed + i
                if i > 0:
                    generator = torch.Generator("cpu").manual_seed(current_seed)
                
                # 스텝 콜백을 위한 상태 저장 (클로저 문제 방지)
                current_image_idx = i
                total_images = num_images
                
                # 스텝별 콜백 함수 생성
                def create_step_callback(img_idx, total_imgs):
                    def step_callback(current_step, total_steps):
                        if progress_callback:
                            # 메인 이벤트 루프에 코루틴 스케줄링
                            asyncio.run_coroutine_threadsafe(
                                progress_callback(img_idx + 1, total_imgs, current_step, total_steps),
                                main_loop
                            )
                    return step_callback
                
                step_cb = create_step_callback(current_image_idx, total_images)
                
                # progress_bar 후킹
                self._hook_progress_bar(step_cb)
                
                try:
                    # 편집 실행 (final_prompt 사용 - 참조 이미지 분석 결과 포함)
                    def run_edit():
                        return self.pipe(
                            image,
                            final_prompt,
                            negative_prompt=negative_prompt,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps,
                            num_images_per_prompt=1,
                            generator=generator
                        ).images[0]
                    
                    result_image = await asyncio.to_thread(run_edit)
                finally:
                    # progress_bar 복원
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
            # 에러 발생 시에도 progress_bar 복원 및 메모리 정리
            self._restore_progress_bar()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            return False, [], f"편집 실패: {str(e)}"


# 전역 인스턴스
longcat_edit_manager = LongCatEditManager()

