import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor

import gradio as gr


# load image
#root = "refined_data/validation/"


#dataset = load_dataset("imagefolder", data_dir=root)#, split="validation")

def predict(image):
    print(image)
    model = torch.load('git_finetune_2.pth',map_location = torch.device('cpu'))
  #  image_read = Image.open(image)
    processor = AutoProcessor.from_pretrained("microsoft/git-base")
   # print(processor)
    #example = dataset[0]
    #image = example["image"]

    width, height = image.size
    image = image.resize((int(0.3*width), int(0.3*height)))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # prepare image for the model
    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs.pixel_values

    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)

    return generated_caption

gr.Interface(fn = predict,
             inputs=gr.Image(type="pil", image_mode="RGB"),
        outputs=gr.Textbox(label="Predicted caption")).launch(share=True, server_port=7860)
