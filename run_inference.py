import argparse
import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from utils.lora import LoRANetwork # Keeping this, assuming lora.py exists in utils/

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

    print(f"Loading Slider from {args.slider_path}...")
    # Initialize LoRA Network. 
    # NOTE: rank=1 is standard for SliderSpace. If you trained with rank=4, change this to rank=4.
    lora_net = LoRANetwork(pipe.unet, rank=1, alpha=1).to(args.device)
    
    try:
        state_dict = torch.load(args.slider_path)
        lora_net.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Tip: If this is a size mismatch, check if you trained with a different Rank.")
        return

    # Apply the slider
    lora_net.apply_to()
    lora_net.set_lora_slider_scale(args.scale)
    
    print(f"Generating: '{args.prompt}' with scale {args.scale}...")
    # Generate
    with torch.no_grad():
        image = pipe(args.prompt, num_inference_steps=50).images[0]

    # Save directly using PIL
    output_filename = f"inference_scale_{args.scale}.png"
    image.save(output_filename)
    print(f"Success! Saved to {output_filename}")

if __name__ == "__main__":
    main()