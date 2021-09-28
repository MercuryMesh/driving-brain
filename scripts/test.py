import sys
from PIL import Image, ImageDraw, ImageColor
import time
from modelplace_api import Device
from pytorch_resnest import InferenceModel
from pycocotools import mask
import random


def run_model(image: Image):
    model = InferenceModel()  # Initialize a model
    model.model_load(Device.cpu)  # Loading a model weights
    ret = model.process_sample(image)  # Processing an image
    return ret


if __name__ == "__main__":
    image = Image.open(sys.argv[1]).convert("RGB")
    mask_color = Image.new("RGB", image.size, color="#ff0000")
    start = time.time()
    ret_val = run_model(image)
    print(f"Model runtime: {time.time() - start}")
    print(ret_val)
    # for rle_mask in ret_val.mask["binary"]:
    # draw = ImageDraw.Draw(image)

    # for rle_mask in zip(ret_val.mask["classes"], ret_val.mask["binary"]):
    #     color = "#" + ''.join([random.choice('ABCDEF0123456789') for i in range(6)])

    #     label = ret_val.classes[rle_mask[0]]
    #     m = mask.decode(rle_mask[1])
    #     bbox = mask.toBbox(rle_mask[1])
        
    #     pil_mask = Image.fromarray(m * 127, "L").convert("1")
    #     c = Image.new("RGB", image.size, color)

    #     draw.rectangle(tuple(bbox), outline=color, width=4)
    #     draw.text((bbox[0], bbox[1]), label)
    #     image.paste(c, mask=pil_mask)
