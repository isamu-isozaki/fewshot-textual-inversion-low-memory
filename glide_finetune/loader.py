import time
from pathlib import Path
from random import randint, choice, random

import PIL

import torch as th
from torch.utils.data import Dataset
from torchvision import transforms as T
from glide_finetune.glide_util import get_tokens_and_mask, get_uncond_tokens_mask
from glide_finetune.train_util import pil_image_to_norm_tensor
import albumentations as A
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import os

def random_resized_crop(image, shape, resize_ratio=1.0):
    """
    Randomly resize and crop an image to a given size.

    Args:
        image (PIL.Image): The image to be resized and cropped.
        shape (tuple): The desired output shape.
        resize_ratio (float): The ratio to resize the image.
    """
    image_transform = T.RandomResizedCrop(shape, scale=(resize_ratio, 1.0), ratio=(1.0, 1.0))
    return image_transform(image)


def get_image_files_dict(base_path):
    image_files = [
        *base_path.glob("**/*.png"),
        *base_path.glob("**/*.jpg"),
        *base_path.glob("**/*.jpeg"),
        *base_path.glob("**/*.bmp"),
    ]
    return {image_file.stem: image_file for image_file in image_files}


def get_text_files_dict(base_path):
    text_files = [*base_path.glob("**/*.txt")]
    return {text_file.stem: text_file for text_file in text_files}


def get_shared_stems(image_files_dict, text_files_dict):
    image_files_stems = set(image_files_dict.keys())
    text_files_stems = set(text_files_dict.keys())
    return list(image_files_stems & text_files_stems)

def normalize_val(val):
    return val/127.5-1

def get_mask_from_input(input_arr, args, tolerance=30):
    inpainting_color = np.array((255, 0, 255))
    if args['grayscale']:
        inpainting_mask = input_arr[..., 0]<normalize_val(tolerance)
    inpainting_tol_colors = normalize_val(np.array((inpainting_color[0]-tolerance, inpainting_color[1]+tolerance, inpainting_color[2]-tolerance)))
    if args['grayscale']:
        inpainting_mask = input_arr[..., 0]<normalize_val(tolerance)
    else:
        inpainting_mask = (input_arr[..., 0]>inpainting_tol_colors[0]) &(input_arr[..., 1] < inpainting_tol_colors[1]) & (input_arr[..., 2]>inpainting_tol_colors[2])
    inpainting_mask = inpainting_mask[..., None]
    return inpainting_mask

imagenet_templates_small = [
    "a photo of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of my {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a good photo of a {}",
    "a photo of the nice {}"
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]

class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
        side_x=64,
        side_y=64,
        resize_ratio=0.75,
        text_ctx_len=128,
        uncond_p=0.0,
        enable_glide_upsample=False,
        upscale_factor=4,
        data_aug=False,
    ):

        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = side_x
        self.side_x=side_x
        self.side_y=side_y
        self.enable_glide_upsample = enable_glide_upsample
        if enable_glide_upsample:
            self.size *= upscale_factor
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.data_aug = data_aug

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        self.num_images = len(self.image_paths)
        self._length = self.num_images
        self.base_transform = A.Compose([
                A.RandomResizedCrop(self.size, self.size, scale=(resize_ratio, 1), ratio=(1, 1), p=1)
            ],
        )
        if set == "train":
            self._length = self.num_images * repeats
        resize_ratio = 0.9
        self.base_transform = A.Compose(
            [
                A.Rotate(p=0.5, limit=10, crop_border=True),
                A.RandomResizedCrop(self.size, self.size, scale=(resize_ratio, 1), ratio=(1, 1), p=0.5)
            ],
        )
        self.conditional_noise = A.Compose(
            [
                A.GaussNoise(p=0.5),
                A.Blur(blur_limit=3, p=0.5),
            ]
        )
        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]

        self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length
    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = choice(self.templates).format(placeholder_string)

        example["input_ids"], example["mask"] = get_tokens_and_mask(tokenizer=self.tokenizer, prompt=text)

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        if self.data_aug:
            transformed = self.base_transform(image=img)
            img = transformed["image"]
        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        if self.enable_glide_upsample:
            # conditional augmentation for base image https://arxiv.org/pdf/2106.15282.pdf
            base_image = image.resize((self.side_x, self.side_y), resample=self.interpolation)
            base_image = self.conditional_noise(np.array(base_image))['image']
            base_image = (base_image / 127.5 - 1.0).astype(np.float32)
            example["base_img"] = th.from_numpy(base_image).permute(2, 0, 1)
        image = (image / 127.5 - 1.0).astype(np.float32)
        example["pixel_values"] = th.from_numpy(image).permute(2, 0, 1)
        if self.enable_glide_upsample:
            return example["input_ids"], example["mask"],example['pixel_values'], example["base_img"]
        else:
            return example["input_ids"], example["mask"], example['pixel_values']

# class TextImageDataset(Dataset):
#     def __init__(
#         self,
#         folder="",
#         side_x=64,
#         side_y=64,
#         resize_ratio=0.75,
#         shuffle=False,
#         tokenizer=None,
#         text_ctx_len=128,
#         uncond_p=0.0,
#         use_captions=False,
#         enable_glide_upsample=False,
#         upscale_factor=4,
#         inpainting=True,
#     ):
    
#         super().__init__()
#         folder = Path(folder)

#         self.image_files = get_image_files_dict(folder)
#         if use_captions:
#             self.text_files = get_text_files_dict(folder)
#             self.keys = get_shared_stems(self.image_files, self.text_files)
#             print(f"Found {len(self.keys)} images.")
#             print(f"Using {len(self.text_files)} text files.")
#         else:
#             self.text_files = None
#             self.keys = list(self.image_files.keys())
#             print(f"Found {len(self.keys)} images.")
#             print(f"NOT using text files. Restart with --use_captions to enable...")
#             time.sleep(3)

#         self.resize_ratio = resize_ratio
#         self.text_ctx_len = text_ctx_len

#         self.shuffle = shuffle
#         self.prefix = folder
#         self.side_x = side_x
#         self.side_y = side_y
#         self.tokenizer = tokenizer
#         self.uncond_p = uncond_p
#         self.enable_upsample = enable_glide_upsample
#         self.upscale_factor = upscale_factor
#         self.transform = A.Compose([
#             A.CLAHE(p=0.5),
#             A.HorizontalFlip(p=0.5),
#             A.RandomRotate90(p=0.5),
#             A.RandomBrightnessContrast(p=0.2),
#         #     A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=[-0.1, 0.3], rotate_limit=20, p=.5, border_mode=cv2.BORDER_CONSTANT, value=(221,213,204)),
#             A.Blur(blur_limit=3, p=0.5),
#             A.ColorJitter(p=0.5),
#             A.Emboss(p=0.5),
#             A.FancyPCA(p=0.5),
#             A.ImageCompression(p=0.5),

#         #     A.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT, value=(221,213,204)),
#         #     A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, value=(221,213,204)),
#         ])
#         if inpainting:
#             self.upsample_transform = A.Compose([
#                     A.RandomResizedCrop(self.side_x * self.upscale_factor, self.side_y * self.upscale_factor, scale=(self.resize_ratio, 1), ratio=(1, 1), p=1)
#                 ],
#                 additional_targets={'image0': 'image'}
#             )
#             self.base_transform = A.Compose([
#                     A.RandomResizedCrop(self.side_x, self.side_y, scale=(self.resize_ratio, 1), ratio=(1, 1), p=1)
#                 ],
#                 additional_targets={'image0': 'image'}
#             )
#         else:
#             self.upsample_transform = A.Compose([
#                     A.RandomResizedCrop(self.side_x * self.upscale_factor, self.side_y * self.upscale_factor, scale=(self.resize_ratio, 1), ratio=(1, 1), p=1)
#                 ],
#             )
#             self.base_transform = A.Compose([
#                     A.RandomResizedCrop(self.side_x, self.side_y, scale=(self.resize_ratio, 1), ratio=(1, 1), p=1)
#                 ],
#             )
#         self.inpainting = inpainting

#     def __len__(self):
#         return len(self.keys)

#     def random_sample(self):
#         return self.__getitem__(randint(0, self.__len__() - 1))

#     def sequential_sample(self, ind):
#         if ind >= self.__len__() - 1:
#             return self.__getitem__(0)
#         return self.__getitem__(ind + 1)

#     def skip_sample(self, ind):
#         if self.shuffle:
#             return self.random_sample()
#         return self.sequential_sample(ind=ind)

#     def get_caption(self, ind):
#         key = self.keys[ind]
#         text_file = self.text_files[key]
#         descriptions = open(text_file, "r").readlines()
#         descriptions = list(filter(lambda t: len(t) > 0, descriptions))
#         try:
#             description = choice(descriptions).strip()
#             return get_tokens_and_mask(tokenizer=self.tokenizer, prompt=description)
#         except IndexError as zero_captions_in_file_ex:
#             print(f"An exception occurred trying to load file {text_file}.")
#             print(f"Skipping index {ind}")
#             return self.skip_sample(ind)

#     def __getitem__(self, ind):
#         key = self.keys[ind]
#         image_file = self.image_files[key]
#         if self.text_files is None or random() < self.uncond_p:
#             tokens, mask = get_uncond_tokens_mask(self.tokenizer)
#         else:
#             tokens, mask = self.get_caption(ind)

#         try:
#             original_pil_image = PIL.Image.open(image_file).convert("RGB")
#             if self.inpainting:
#                 imgs = np.array(original_pil_image)
#                 w = imgs.shape[0]
#                 masked_image, original_pil_image = imgs[:w, :], imgs[w:, :]
#                 mask = get_mask_from_input(masked_image)
#         except (OSError, ValueError) as e:
#             print(f"An exception occurred trying to load file {image_file}.")
#             print(f"Skipping index {ind}")
#             return self.skip_sample(ind)
#         if self.enable_upsample: # the base image used should be derived from the cropped high-resolution image.
#             if self.inpainting:
#                 transformed = self.upsample_transform(image=original_pil_image, image0=mask)
#                 upsample_pil_image, upsample_mask = transformed["image"], transformed["image0"]
#                 upsample_mask = get_mask_from_input(upsample_mask)
#                 upsample_mask = th.from_numpy(upsample_mask).float().permute(2, 0, 1)
#             else:
#                 transformed = self.upsample_transform(image=original_pil_image)
#                 upsample_pil_image = PIL.Image.fromArray(transformed["image"])
#                 upsample_mask = None
#             upsample_tensor = pil_image_to_norm_tensor(upsample_pil_image)
#             base_pil_image = upsample_pil_image.resize((self.side_x, self.side_y), resample=PIL.Image.BICUBIC)
#             base_tensor = pil_image_to_norm_tensor(base_pil_image)
#             return th.tensor(tokens), th.tensor(mask, dtype=th.bool), base_tensor, upsample_tensor, upsample_mask
#         if self.inpainting:
#             transformed = self.base_transform(image=original_pil_image, image0=mask)
#             base_pil_image, base_mask = transformed["image"], transformed["image0"]
#             base_mask = get_mask_from_input(base_mask)
#             base_mask = th.from_numpy(base_mask).float().permute(2, 0, 1)
#         else:
#             transformed = self.base_transform(image=original_pil_image)
#             base_pil_image = transformed["image"]
#             base_mask = None    
#         base_tensor = pil_image_to_norm_tensor(base_pil_image)
#         return th.tensor(tokens), th.tensor(mask, dtype=th.bool), base_tensor, base_mask