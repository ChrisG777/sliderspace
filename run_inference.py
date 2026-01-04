import argparse
import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from utils.utils import save_images 
from utils.lora import LoRANetwork, lora_forward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a picture of a spaceship")
    parser.add_argument("--slider_path", type=str, required=True, help="Path to the .pt file")
    parser.add_argument("--scale", type=float, default=2.0, help="Slider strength (can be negative!)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # 1. Load SDXL Base
    print("Loading SDXL...")
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(args.device)

    # 2. Load the Slider (LoRA)
    print(f"Loading Slider from {args.slider_path}...")
    lora_net = LoRANetwork(pipe.unet, rank=1, alpha=1).to(args.device)
    state_dict = torch.load(args.slider_path)
    lora_net.load_state_dict(state_dict)

    # 3. Inject the LoRA into the UNet
    # This is the "magic" part where the slider takes effect
    lora_net.apply_to()

    # 4. Generate Image
    print(f"Generating: '{args.prompt}' with scale {args.scale}...")
    # Note: SliderSpace usually hacks the LoRA scale manually or via the forward pass.
    # We set the 'multiplier' to control strength.
    lora_net.set_lora_slider_scale(args.scale)
    
    image = pipe(args.prompt, num_inference_steps=50).images[0]

    # 5. Save
    output_filename = f"inference_{args.scale}.png"
    image.save(output_filename)
    print(f"Saved to {output_filename}")

if __name__ == "__main__":
    main()