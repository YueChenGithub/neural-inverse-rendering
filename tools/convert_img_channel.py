from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import torch

ToTensor = ToTensor()
ToPILImage = ToPILImage()

def rgba2rgb(img:Image.Image, background=0):
    """
    Convert RGBA image to RGB image.
    """
    if img.mode == 'RGBA':
        img = ToTensor(img)
        color = img[:3, :, :]
        mask = img[3, :, :]
        img = color * mask[None, :, :] + background * (1 - mask[None, :, :])
    else:
        img = ToTensor(img)
    return img


def rgba2rgb_pil(img:Image.Image, background=0):
    """
    Convert RGBA image to RGB image.
    """
    assert img.mode == 'RGBA'
    img = ToTensor(img)
    color = img[:3, :, :]
    mask = img[3, :, :]
    img = color * mask[None, :, :] + background * (1 - mask[None, :, :])
    img = ToPILImage(img)
    return img

def rgb2rgba(rgb:Image.Image, rgba_ref:Image.Image, background=0):
    """
    Convert RGB image to RGBA image.
    """
    assert rgb.mode == 'RGB'
    assert rgba_ref.mode == 'RGBA'
    rgb = ToTensor(rgb)
    rgba_ref = ToTensor(rgba_ref)
    mask = rgba_ref[3, :, :]
    mask = mask[None, :, :]

    mask_reverse = torch.where(mask==0, 0, 1 / mask)
    color = (rgb - background) * mask_reverse + background
    color = torch.clamp(color, 0, 1)
    img = torch.cat([color, mask], dim=0)
    img = ToPILImage(img)
    return img

def rgb_change_background(rgb:Image.Image, rgba_ref:Image.Image, background_origin=0, background_new=1):
    # change background color from background_origin to background_new
    rgb_a = rgb2rgba(rgb, rgba_ref, background=background_origin)
    rgb_background = rgba2rgb_pil(rgb_a, background=background_new)
    return rgb_background

if "__main__" == __name__:
    rgba_path = "/home/yue/Desktop/test_000/rgba.png"
    rgba = Image.open(rgba_path)

    rgb_path = "/home/yue/Desktop/test_000/DiffCol_0001.png"
    rgb = Image.open(rgb_path).convert("RGB")  # rgb[:, :, :3]

    # image = rgba2rgb(rgba, 0.5)
    # ToPILImage(image).show()



    image = rgb_change_background(rgb, rgba)
    # ref = rgba2rgb_pil(rgba, background=1)

    # diff = ToTensor(image) - ToTensor(ref)
    # print(diff.max(), diff.min())
    # diff = torch.abs(diff)
    # ToPILImage(diff).show()

    image.show()
    # ref.show()

