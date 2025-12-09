"""Z-Image Diffusers Inference."""

import os
import torch
from diffusers import ZImagePipeline

# GGUF 양자화 옵션 (None = BF16 기본, 또는 양자화 타입 선택)
# 사용 가능: Q8_0, Q6_K, Q5_K_M, Q5_K_S, Q4_K_M, Q4_K_S, Q3_K_M, Q3_K_S
GGUF_QUANTIZATION = None  # 예: "Q4_K_M" 으로 설정하면 GGUF 모델 사용

# GGUF 파일명 매핑
GGUF_FILES = {
    "Q8_0": "z_image_turbo-Q8_0.gguf",
    "Q6_K": "z_image_turbo-Q6_K.gguf",
    "Q5_K_M": "z_image_turbo-Q5_K_M.gguf",
    "Q5_K_S": "z_image_turbo-Q5_K_S.gguf",
    "Q4_K_M": "z_image_turbo-Q4_K_M.gguf",
    "Q4_K_S": "z_image_turbo-Q4_K_S.gguf",
    "Q3_K_M": "z_image_turbo-Q3_K_M.gguf",
    "Q3_K_S": "z_image_turbo-Q3_K_S.gguf",
}


def load_pipeline(device: str, use_gguf: str = None):
    """
    Z-Image 파이프라인 로드
    
    Args:
        device: 사용할 디바이스 (cuda, mps, cpu)
        use_gguf: GGUF 양자화 타입 (None이면 BF16 사용)
    
    Returns:
        ZImagePipeline 인스턴스
    """
    if use_gguf and use_gguf in GGUF_FILES:
        # GGUF 양자화 모델 로드
        from diffusers import ZImageTransformer2DModel, GGUFQuantizationConfig
        from huggingface_hub import hf_hub_download
        
        print(f"Loading GGUF model ({use_gguf})...")
        
        # GGUF 파일 다운로드
        gguf_path = hf_hub_download(
            repo_id="jayn7/Z-Image-Turbo-GGUF",
            filename=GGUF_FILES[use_gguf]
        )
        
        # GGUF Transformer 로드
        transformer = ZImageTransformer2DModel.from_single_file(
            gguf_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
        )
        
        # 파이프라인 구성
        pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
    else:
        # 기본 BF16 모델 로드
        print("Loading Z-Image-Turbo model (BF16)...")
        pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )
    
    pipe.to(device)
    return pipe


def main():
    # 설정
    output_dir = "outputs"       # 출력 폴더
    num_images = 10               # 생성할 이미지 수
    height = 512
    width = 512
    num_inference_steps = 9      # 실제로 8 DiT forward
    guidance_scale = 0.0         # Turbo 모델은 0으로 설정
    seed = 42                    # 시작 시드 (각 이미지마다 +1 증가)
    
    prompt = (
        "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. "
        "Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. "
        "Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, "
        "silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
    )

    prompt = "A hyper-realistic portrait of a beautiful young Asian woman with a fair, porcelain complexion. She has long, voluminous dark wavy hair parted in the middle, framing her face softly. She is wearing a light gray ribbed knit top with a square neckline that highlights her collarbones, paired with a delicate thin silver necklace. Her makeup is natural and soft, with rosy cheeks and glossy pink lips. She stands against a clean white wall background under soft, diffused studio lighting. The image is high-quality, sharp, and photorealistic."

    prompt = "A split-image composition: on the left, a dense, rain-slicked cyberpunk cityscape at night with complex neon reflections; on the right, a serene, sun-drenched Korean zen garden with a single cherry tree in full bloom. The dividing line is sharp and clean. Intricate detail, **Z-Image Turbo** realism, **Z-Image** prompt adherence test."

    prompt = "[Product: Leather handbag] Floating centered against pure white background (hex #FFFFFF), soft drop shadow, 45-degree angle view, studio lighting setup || Details: Clear stitching visibility, hardware details in focus, leather texture prominent || Technical: 2000x2000px, 300 DPI, sharp product edges, commercial photography style || Consistency: Maintain exact position, lighting, and shadow angle across multiple products || Additional: Include clipping path"

    prompt = "A high-quality digital illustration in the style of a Korean webnovel cover. A fierce, handsome East Asian emperor stands amidst a burning palace courtyard at night. He has black hair and is wearing a royal red traditional robe (Hanbok) with intricate gold dragon embroidery. His face and clothes are stained with blood, and he holds a long sword in his hand, looking forward with an intense, angry expression, gritting his teeth. The background features traditional tiled-roof buildings engulfed in roaring flames, smoke, and a dark starry sky with a bright full moon. Red plum blossom branches hang in the foreground, framing the scene. Cinematic lighting, dramatic fire glow illuminating the character, high contrast, semi-realistic anime style, 8k resolution, detailed texture."

    prompt = ""

    # 디바이스 선택
    if torch.cuda.is_available():
        device = "cuda"
        print("Chosen device: cuda")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Chosen device: mps")
    else:
        device = "cpu"
        print("Chosen device: cpu")

    # 파이프라인 로드 (GGUF 양자화 옵션에 따라)
    pipe = load_pipeline(device, GGUF_QUANTIZATION)
    print("Model loaded!")

    # [선택] CPU 오프로딩 (VRAM이 부족한 경우 주석 해제)
    # pipe.enable_model_cpu_offload()

    # 출력 폴더 생성
    os.makedirs(output_dir, exist_ok=True)

    # 이미지 생성
    print(f"Generating {num_images} images...")
    for i in range(num_images):
        current_seed = seed + i
        print(f"\n[{i+1}/{num_images}] Generating with seed {current_seed}...")
        
        image = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device).manual_seed(current_seed),
        ).images[0]

        output_path = os.path.join(output_dir, f"image_{i+1:03d}_seed{current_seed}.png")
        image.save(output_path)
        print(f"Saved: {output_path}")
    
    print(f"\nDone! {num_images} images saved to '{output_dir}/' folder.")


if __name__ == "__main__":
    main()
