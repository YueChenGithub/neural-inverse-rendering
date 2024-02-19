import torch
from pathlib import Path
import torchvision

def linear2srgb(f: torch.Tensor) -> torch.Tensor:
    srgb = torch.where(f <= 0.0031308, f * 12.92, torch.pow(torch.clamp(f, 0.0031308), 1.0 / 2.4) * 1.055 - 0.055)
    return torch.clamp(srgb, 0, 1.0)


def linear2srgb_naive(color):
    srgb_exponent = 2.2  # blender gamma = 2.2
    color = torch.clamp(color, 1e-4, 1)
    color = torch.pow(color, 1 / srgb_exponent)
    return color


def srgb2linear(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f <= 0.04045, f / 12.92, torch.pow((torch.clamp(f, 0.04045) + 0.055) / 1.055, 2.4))


def X2PIL(image, tone_mapping=True):
    if type(image) is not torch.Tensor:
        try:
            image = image.torch()
        except:
            print('image can not be converted into torch')

    assert len(image.shape) == 3, 'image dimension not correct'

    """change to format [C,H,W]"""
    if image.shape[0] not in [1, 3, 4]:
        image = image.permute(2, 0, 1)
    """tone mapping"""
    if tone_mapping:
        image = linear2srgb(image)
    else:
        image = torch.clamp(image, 0, 1.0)
    ToPILImage = torchvision.transforms.ToPILImage()
    image = ToPILImage(image)
    return image


def show_image(image, tone_mapping=True):
    image = X2PIL(image, tone_mapping=tone_mapping)
    image.show()


def save_image(image, save_image_path, tone_mapping=True):
    image = X2PIL(image, tone_mapping=tone_mapping)
    Path(save_image_path).parent.mkdir(exist_ok=True, parents=True)
    image.save(save_image_path)
