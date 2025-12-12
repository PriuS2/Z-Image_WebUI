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
    EDIT_GPU_INDEX,
    EDIT_TEXT_ENCODER_GPU,
    EDIT_TRANSFORMER_GPU,
)
from utils.llm_client import llm_client


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
        self.current_model: Optional[str] = None
        self.device: Optional[str] = None
        self.gpu_index: int = EDIT_GPU_INDEX
        # 컴포넌트별 GPU 설정 (-1이면 분산 비활성화)
        self.text_encoder_gpu: int = EDIT_TEXT_ENCODER_GPU
        self.transformer_gpu: int = EDIT_TRANSFORMER_GPU  # VAE도 함께 배치
        self.distributed_mode: bool = False  # 분산 모드 활성화 여부
        self._lock = asyncio.Lock()
    
    @property
    def is_loaded(self) -> bool:
        """모델 로드 여부"""
        return self.pipe is not None
    
    def get_device(self, gpu_index: Optional[int] = None) -> str:
        """사용 가능한 디바이스 반환 (멀티 GPU 지원)"""
        if gpu_index is None:
            gpu_index = self.gpu_index
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_index >= gpu_count:
                gpu_index = 0  # 유효하지 않으면 0으로 폴백
            return f"cuda:{gpu_index}"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    async def load_model(
        self,
        quantization: str = "BF16 (기본, 최고품질)",
        cpu_offload: bool = True,
        model_path: Optional[str] = None,
        gpu_index: Optional[int] = None,
        text_encoder_gpu: Optional[int] = None,
        transformer_gpu: Optional[int] = None,
        progress_callback: Optional[Callable[[int, str, str], Any]] = None
    ) -> Tuple[bool, str]:
        """
        LongCat-Image-Edit 모델 로드
        
        Args:
            quantization: 양자화 옵션
            cpu_offload: CPU 오프로딩 사용 여부 (VRAM 절약)
            model_path: 커스텀 모델 경로
            gpu_index: 사용할 GPU 인덱스 (None이면 기본값 사용) - 분산 비활성화 시 사용
            text_encoder_gpu: Text Encoder GPU 인덱스 (-1 또는 None이면 분산 안함)
            transformer_gpu: Transformer + VAE GPU 인덱스 (-1 또는 None이면 분산 안함)
            progress_callback: 진행상황 콜백 (percent, label, detail)
        
        Returns:
            (success, message)
        """
        async with self._lock:
            if self.pipe is not None:
                return False, "모델이 이미 로드되어 있습니다. 먼저 언로드하세요."
            
            try:
                # GPU 인덱스 설정
                if gpu_index is not None:
                    self.gpu_index = gpu_index
                
                # 컴포넌트별 GPU 설정 (-1 또는 None이면 분산 비활성화)
                if text_encoder_gpu is not None and text_encoder_gpu >= 0:
                    self.text_encoder_gpu = text_encoder_gpu
                if transformer_gpu is not None and transformer_gpu >= 0:
                    self.transformer_gpu = transformer_gpu
                
                # GPU 인덱스 유효성 검사
                gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
                if gpu_count > 0 and self.gpu_index >= gpu_count:
                    self.gpu_index = 0  # 유효하지 않으면 0으로 폴백
                
                # 분산 모드 활성화 여부 확인 (Text Encoder와 Transformer+VAE를 다른 GPU에 배치)
                self.distributed_mode = (
                    gpu_count > 1 and 
                    (self.text_encoder_gpu >= 0 or self.transformer_gpu >= 0)
                )
                
                self.device = self.get_device(self.gpu_index)
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
                
                # 분산 모드 메시지
                if self.distributed_mode:
                    dist_info = []
                    if self.text_encoder_gpu >= 0:
                        dist_info.append(f"TextEnc→GPU{self.text_encoder_gpu}")
                    if self.transformer_gpu >= 0:
                        dist_info.append(f"Trans+VAE→GPU{self.transformer_gpu}")
                    report_progress(5, "🔧 분산 모드로 모델 초기화 중...", ", ".join(dist_info))
                else:
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
                gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
                
                if self.distributed_mode and gpu_count > 1:
                    # 분산 모드: 각 컴포넌트를 지정된 GPU로 이동 + accelerate hooks 사용
                    report_progress(85, "🔀 컴포넌트별 GPU 분산 중...", "")
                    
                    def distribute_components():
                        from accelerate.hooks import add_hook_to_module, AlignDevicesHook
                        
                        default_device = torch.device(f"cuda:{self.gpu_index}")
                        
                        # Text Encoder 배치
                        te_device = default_device
                        if self.text_encoder_gpu >= 0 and self.text_encoder_gpu < gpu_count:
                            te_device = torch.device(f"cuda:{self.text_encoder_gpu}")
                        if hasattr(self.pipe, 'text_encoder') and self.pipe.text_encoder is not None:
                            self.pipe.text_encoder = self.pipe.text_encoder.to(te_device)
                            # Hook 추가: forward 시 입력을 자동으로 해당 디바이스로 이동
                            add_hook_to_module(self.pipe.text_encoder, AlignDevicesHook(execution_device=te_device))
                            print(f"📍 Text Encoder → {te_device}")
                        
                        # Transformer + VAE 배치 (같은 GPU에 함께 배치)
                        tf_vae_device = default_device
                        if self.transformer_gpu >= 0 and self.transformer_gpu < gpu_count:
                            tf_vae_device = torch.device(f"cuda:{self.transformer_gpu}")
                        
                        if hasattr(self.pipe, 'transformer') and self.pipe.transformer is not None:
                            self.pipe.transformer = self.pipe.transformer.to(tf_vae_device)
                            add_hook_to_module(self.pipe.transformer, AlignDevicesHook(execution_device=tf_vae_device))
                            print(f"📍 Transformer → {tf_vae_device}")
                        
                        # VAE는 Transformer와 같은 GPU에 배치
                        if hasattr(self.pipe, 'vae') and self.pipe.vae is not None:
                            self.pipe.vae = self.pipe.vae.to(tf_vae_device)
                            add_hook_to_module(self.pipe.vae, AlignDevicesHook(execution_device=tf_vae_device))
                            print(f"📍 VAE → {tf_vae_device} (Transformer와 동일)")
                    
                    await asyncio.to_thread(distribute_components)
                    
                    # 분산 배치 정보 생성
                    dist_info = []
                    if self.text_encoder_gpu >= 0:
                        dist_info.append(f"TextEnc→GPU{self.text_encoder_gpu}")
                    if self.transformer_gpu >= 0:
                        dist_info.append(f"Trans+VAE→GPU{self.transformer_gpu}")
                    
                    report_progress(95, "⚙️ 분산 배치 완료", ", ".join(dist_info) if dist_info else "기본 GPU 사용")
                
                elif cpu_offload:
                    # CPU 오프로딩 모드
                    gpu_name = torch.cuda.get_device_properties(self.gpu_index).name if gpu_count > 0 else "N/A"
                    report_progress(85, f"🚀 GPU{self.gpu_index} ({gpu_name})로 모델 전송 중...", "")
                    await asyncio.to_thread(self.pipe.enable_model_cpu_offload, gpu_id=self.gpu_index)
                    report_progress(95, "⚙️ CPU 오프로딩 설정됨", f"GPU{self.gpu_index} 사용, VRAM 부족 시 RAM 사용")
                
                else:
                    # 단일 GPU 모드
                    gpu_name = torch.cuda.get_device_properties(self.gpu_index).name if gpu_count > 0 else "N/A"
                    report_progress(85, f"🚀 GPU{self.gpu_index} ({gpu_name})로 모델 전송 중...", "")
                    await asyncio.to_thread(self.pipe.to, self.device, torch.bfloat16)
                
                self.current_model = quantization
                
                # 완료 메시지
                if self.distributed_mode:
                    report_progress(100, "✅ 분산 모드로 모델 로드 완료!", self._get_all_vram_info())
                else:
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
        """현재 GPU의 VRAM 사용량 정보"""
        if torch.cuda.is_available():
            gpu_idx = self.gpu_index
            vram_used = torch.cuda.memory_allocated(gpu_idx) / 1024**3
            vram_total = torch.cuda.get_device_properties(gpu_idx).total_memory / 1024**3
            return f"GPU{gpu_idx} VRAM: {vram_used:.1f}GB / {vram_total:.1f}GB"
        return "N/A"
    
    def _get_all_vram_info(self) -> str:
        """모든 GPU의 VRAM 사용량 정보 (분산 모드용)"""
        if torch.cuda.is_available():
            infos = []
            gpu_count = torch.cuda.device_count()
            
            # 사용 중인 GPU만 표시
            used_gpus = set([self.gpu_index])
            if self.text_encoder_gpu >= 0:
                used_gpus.add(self.text_encoder_gpu)
            if self.transformer_gpu >= 0:
                used_gpus.add(self.transformer_gpu)  # VAE도 여기에 포함
            
            for gpu_idx in sorted(used_gpus):
                if gpu_idx < gpu_count:
                    vram_used = torch.cuda.memory_allocated(gpu_idx) / 1024**3
                    vram_total = torch.cuda.get_device_properties(gpu_idx).total_memory / 1024**3
                    infos.append(f"GPU{gpu_idx}: {vram_used:.1f}/{vram_total:.1f}GB")
            
            return " | ".join(infos)
        return "N/A"
    
    def _hook_progress_bar(self, step_callback):
        """파이프라인의 progress_bar를 후킹하여 스텝별 콜백 호출"""
        pipe = self.pipe
        original_progress_bar = pipe.progress_bar.__func__  # 언바운드 메서드 가져오기
        
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
        
        return original_progress_bar
    
    def _restore_progress_bar(self, original_progress_bar):
        """원래 progress_bar 복원"""
        import types
        if self.pipe and original_progress_bar:
            self.pipe.progress_bar = types.MethodType(original_progress_bar, self.pipe)
    
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
                original_progress_bar = self._hook_progress_bar(step_cb)
                
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

