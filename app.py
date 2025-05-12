import gradio as gr
import spaces
from diffusers import StableDiffusionPipeline

MODEL_OPTIONS = {
    "Stable Diffusion 2.1": "stabilityai/stable-diffusion-2-1",
    "OpenJourney": "prompthero/openjourney",
    "Waifu Diffusion (NSFW)": "hakurei/waifu-diffusion"
}

@spaces.GPU
def generate_image(prompt, model_name):
    model_id = MODEL_OPTIONS[model_name]
    pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None)
    pipe = pipe.to("cuda")
    image = pipe(prompt).images[0]
    return image

iface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Dropdown(list(MODEL_OPTIONS.keys()), label="Model", value="Waifu Diffusion (NSFW)")
    ],
    outputs=gr.Image(type="pil"),
    title="ZeroGPU Text-to-Image Generator (NSFW/Unfiltered)"
)

if __name__ == "__main__":
    iface.launch()
