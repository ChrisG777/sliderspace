import argparse
import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a picture of a spaceship")
    parser.add_argument("--slider_path", type=str, required=True, help="Path to the .pt file")
    parser.add_argument("--scale", type=float, default=2.0, help="Slider strength")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print(f"Loading SDXL on {args.device}...")
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(args.device)

    print(f"Loading LoRA from {args.slider_path}...")
    
    # NATIVE LOADING: This handles the key mapping automatically
    try:
        pipe.load_lora_weights(args.slider_path, adapter_name="slider")
    except Exception as e:
        print(f"Native load failed: {e}")
        print("Trying stricter loading...")
        # Fallback for some edge cases in saved weights
        state_dict = torch.load(args.slider_path)
        pipe.load_lora_weights(state_dict, adapter_name="slider")

    # Set the scale (Strength of the slider)
    # The 'adapter_name' allows us to control the strength of this specific LoRA
    pipe.set_adapters(["slider"], adapter_weights=[args.scale])

    print(f"Generating: '{args.prompt}' with scale {args.scale}...")
    
    image = pipe(args.prompt, num_inference_steps=50).images[0]

    # Clean filename
    clean_scale = str(args.scale).replace(".", "p")
    output_filename = f"inference_{clean_scale}.png"
    
    image.save(output_filename)
    print(f"\nSUCCESS! Image saved to: {output_filename}")

if __name__ == "__main__":
    main()