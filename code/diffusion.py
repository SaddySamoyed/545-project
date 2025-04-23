

#@title stable diffusion
model_id = "runwayml/stable-diffusion-inpainting"
pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)


# load image
# Load your input image
input_image_path = "image.png"  # Replace with your image file path
image = Image.open(input_image_path).convert("RGB")

outputfolder = "OutputFolder"
os.makedirs(outputfolder, exist_ok=True)


for label in label_detected:
    mask_path = f"mask_{label}.png"
    mask_path = os.path.join(maskFolder, mask_path)
    mask = Image.open(mask_path).convert("RGB")
    output_image_path = f"output_{label}.png"
    output_image_path = os.path.join(outputfolder, output_image_path)

    result = pipe(prompt=prompt, image=image, mask_image=mask).images[0]

    result.save(output_image_path)