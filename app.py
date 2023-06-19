#Created for https://www.aiqrgenerator.com/ as a public beta version for embedding. 
#I wanted to make the model more accessable for public users and commercialized. Feel free to share at https://www.aiqrgenerator.com/generator.
#May update again but will probably remain the final public version as I am still working on features and consider this a minimum viable product
#Further updates and custom models will be updated privately.



#derivative and edited from QR-code-AI-art-generator by patrickvonplaten - customized AND COPYRIGHTED UNDER COMMERCIAL LICENSE
#ControlNet model is controlnet_qrcode-control_v1p_sd15 by DionTimmer from under OPENRAIL license
#to do - remove stable diff 2 API and use my custom model for generation for init image
#add init image !!!
#custom controlnetmodel implementation
#V1.02 public, not recent version




import torch
import gradio as gr
from PIL import Image
import qrcode
from pathlib import Path
from multiprocessing import cpu_count
import requests
import io
import os
from PIL import Image

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DEISMultistepScheduler,
    HeunDiscreteScheduler,
    EulerDiscreteScheduler,
)

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
HF_TOKEN = os.environ.get("HF_TOKEN")

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content

qrcode_generator = qrcode.QRCode(
    version=1,
    error_correction=qrcode.ERROR_CORRECT_H,
    box_size=10,
    border=4,
)

controlnet = ControlNetModel.from_pretrained(
    "DionTimmer/controlnet_qrcode-control_v1p_sd15", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16,
).to("cuda")
pipe.enable_xformers_memory_efficient_attention()



def resize_for_condition_image(input_image: Image.Image, resolution: int = 512):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 32.0)) * 32
    W = int(round(W / 32.0)) * 32
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img


SAMPLER_MAP = {
    "DPM++ Karras SDE": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True, algorithm_type="sde-dpmsolver++"),
    "DPM++ Karras": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True),
    "Heun": lambda config: HeunDiscreteScheduler.from_config(config),
    "Euler": lambda config: EulerDiscreteScheduler.from_config(config),
    "DDIM": lambda config: DDIMScheduler.from_config(config),
    "DEIS": lambda config: DEISMultistepScheduler.from_config(config),
}


def inference(
    qr_code_content: str,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float = 10.0,
    controlnet_conditioning_scale: float = 2.0,
    strength: float = 0.8,
    seed: int = -1,
    init_image: Image.Image | None = None,
    qrcode_image: Image.Image | None = None,
    use_qr_code_as_init_image = True,
    sampler = "DDIM",
):
    if prompt is None or prompt == "":
        raise gr.Error("Prompt is required")

    if qrcode_image is None and qr_code_content == "":
        raise gr.Error("QR Code Image or QR Code Content is required")

    pipe.scheduler = SAMPLER_MAP[sampler](pipe.scheduler.config)

    generator = torch.manual_seed(seed) if seed != -1 else torch.Generator()

    if qr_code_content != "" or qrcode_image.size == (1, 1):
        print("Generating QR Code from content")
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=4,
        )
        qr.add_data(qr_code_content)
        qr.make(fit=True)

        qrcode_image = qr.make_image(fill_color="black", back_color="white")
        qrcode_image = resize_for_condition_image(qrcode_image, 512)
    else:
        print("Using QR Code Image")
        qrcode_image = resize_for_condition_image(qrcode_image, 512)

    # hack due to gradio examples
    if use_qr_code_as_init_image:
        init_image = qrcode_image
    elif init_image is None or init_image.size == (1, 1):
        print("Generating random image from prompt using Stable Diffusion 2.1 via Inference API")
        # generate image from prompt
        image_bytes = query({"inputs": prompt})
        init_image = Image.open(io.BytesIO(image_bytes))
    else:
        print("Using provided init image")
        init_image = resize_for_condition_image(init_image, 512)

    #promptstart = ""
    promptend = ", high quality, high resolution"
    prompt += promptend 

    negative_promptend = ", butt, nipple, nsfw, nude, nudity, naked"
    negative_prompt += negative_promptend
    
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=qrcode_image,
        control_image=qrcode_image,  # type: ignore
        width=512,  # type: ignore
        height=512,  # type: ignore
        guidance_scale=float(guidance_scale),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),  # type: ignore
        generator=generator,
        strength=float(strength),
        num_inference_steps=25,
    )
    return out.images[0]  # type: ignore

#removed text 
with gr.Blocks() as blocks:
    gr.Markdown(
        """
                                            # CREATED FOR  HTTPS://WWW.AIQRGENERATOR.COM/ EARLY BETA PUBLIC ACCESS V1.02    
                                
==================================**DISCLAIMER - By using this model you agree to waive any liability and are assuming all responsibility for generated images.**===================================    
====================================================================**This model is not intended for commerical use.**==============================================================================    
This generator is trained using SD 1.5. To use SD 2.1 for better quality and other features like upscaling, personal images, dynamic QR codes, style options, and more:
check out our newest model at https://www.aiqrgenerator.com/pro-model
  
When sharing generated QR codes generated with this specific model, please credit aiqrgenerator.com. Feel free to embbed the model or share a link to the website page.

Type in what you want the QR code to look like. Use major subjects seperated by commas like the example below - you can even include styles! 
Type your QR code information such as a website link or if you have a QR image, upload it.  
Feel free to test custom settings as well to make the QR work or try changing your prompt. Change the seed to any number to completely change your generation.   
**Hit run!**  
   
   
==============================================================================================================================================================================    


                """
    )
    prompt = gr.Textbox(
    label="Prompt",
    info="Input subjects or styles you want to see that describes your image - Ex. Mountain, snow, morning, art, painting, digital",
    )

    negative_prompt = gr.Textbox(visible=True, label="Negative Prompt", 
    info="Input things you don't want to see in your image for the model.",
    value="poorly drawn, blurry image, deformed, low resolution, disfigured, low quality, blurry")

    with gr.Row():
        with gr.Column():
            qr_code_content = gr.Textbox(
                label="QR Code Content",
                info="QR Code Content or URL",
                value="https://www.aiqrgenerator.com/",
            )
            with gr.Accordion(label="QR Code Image (Optional)", open=False):
                qr_code_image = gr.Image(
                    label="QR Code Image (Optional). Leave blank to automatically generate QR code",
                    type="pil",
                )

            #negative_prompt = gr.Textbox(
            #    label="Negative Prompt",
            #    value="disfigured, low quality, blurry, nsfw",
            #)

            use_qr_code_as_init_image = gr.Checkbox(visible= False, label="QR Code is used as initial image.", value=True, interactive=False, info="Whether init image should be QR code. Unclick to pass init image or generate init image with Stable Diffusion 2.1")

            with gr.Accordion(label="Init Images (Optional)", open=False, visible=False) as init_image_acc:
                init_image = gr.Image(visible=False, label="Init Image (Optional). Leave blank to generate image with SD 2.1", type="pil")

            #def change_view(qr_code_as_image: bool):
            #    if not qr_code_as_image:
            #        return {init_image_acc: gr.update(visible=True)}
            #    else:
            #        return {init_image_acc: gr.update(visible=False)}

            #use_qr_code_as_init_image.change(change_view, inputs=[use_qr_code_as_init_image], outputs=[init_image_acc])

            with gr.Accordion(
                label="You can modify the generation slightly using the below sliders. See details below. \n ",
                open=True,
            ):
                controlnet_conditioning_scale = gr.Slider(
                    minimum=0.6,
                    maximum=2.0,
                    step=0.01,
                    value=1.00,
                    label="QR High Pass",
                )
                strength = gr.Slider(
                    minimum=0.8, maximum=.95, step=0.01, value=0.9, label="QR Initial Weight"
                )
                guidance_scale = gr.Slider(
                    minimum=5.0,
                    maximum=15.0,
                    step=0.25,
                    value=8.0,
                    label="Prompt Weight",
                )
                sampler = gr.Textbox(visible=False, value="DDIM") #gr.Dropdown(choices=list(SAMPLER_MAP.keys()), value="DPM++ Karras SDE")
                seed = gr.Slider(
                    minimum=-1,
                    maximum=9999999999,
                    step=1,
                    value=2313123,
                    label="Seed",
                    randomize=True,
                )
            with gr.Row():
                run_btn = gr.Button("Run")
        with gr.Column():
            result_image = gr.Image(label="Result Image")
    run_btn.click(
        inference,
        inputs=[
            qr_code_content,
            prompt,
            negative_prompt,
            guidance_scale,
            controlnet_conditioning_scale,
            strength,
            seed,
            init_image,
            qr_code_image,
            use_qr_code_as_init_image,
            sampler,
        ],
        outputs=[result_image],
    )
    gr.Markdown(
        """
### Settings Details
**QR High Pass** - Change this to affect how much the QR code is overlayed to your image in a second pass. Controlnet model.
(Higher setting is more QR code, lower setting is less QR code.)
   
**QR Initial Weight** - Change this to affect how much your image starts looking like a QR code!   
(Higher settings mean your image starts with less QR, lower means the QR will appear sharper)
  
**Prompt Weight** - This determines how much the AI "Listens" to your prompt and try to put what you described into your image.  
(Lower means it is more absract and higher follows your directions more.)
  
**Seed** - This is a randomizer! Use the same seed to generate the same image over and over. Change the seed to change up your image!
(You can copy your seed from a previous generation to get the same image.)
                """
    )

blocks.queue(concurrency_count=1, max_size=20)
blocks.launch(share=False)