import gc
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import torch.nn.functional as F

GAMMA = 1.5
THRED_E = 0.05
THRED_S = 0.3
POOLING_KERNAL_SIZE = 5
DATASET_PATH = '/datasets/path'
DATASETS = ['mipnerf360', 'blending', 'tandt', 'bungeenerf']
SCENES = \
{
    'mipnerf360':['bicycle', 'bonsai', 'counter', 'kitchen', 'room', 'stump', 'garden', 'flowers', 'treehill'],
    'blending':['drjohnson', 'playroom'],
    'tandt':['train', 'truck'],
    'bungeenerf':['amsterdam', 'barcelona', 'bilbao', 'chicago', 'hollywood', 'pompidou', 'quebec', 'rome'],
}

def gamma_correction(image):
    gamma = GAMMA
    return image.point(lambda x: 255 * (x / 255) ** (1 / gamma))

def save_image_L(image, source_path, dir, file_name):
    save_path = os.path.join(source_path, dir)
    os.makedirs(save_path, exist_ok=True)

    H, W = image.shape[-2], image.shape[-1]
    image = (image * 255).clamp(0, 255).byte()
    data_np = image.reshape(H, W).detach().cpu().numpy()

    image = Image.fromarray(data_np)
    image = image.convert('L')
    image.save(f"{save_path}/{file_name}.png")

def load_image_L(path, gamma=True):
    image = Image.open(path).convert("L")

    if gamma:
        image = gamma_correction(image)

    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0).cuda()

    return image

def load_images(source_path, gamma=True, save_l=False):
    dir_path = os.path.join(source_path, "images")
    path = os.listdir(dir_path)
    files = [f for f in path if os.path.isfile(os.path.join(dir_path, f))]
    images = {}

    for idx, file in enumerate(files):
        sys.stdout.write('\r')
        sys.stdout.write("Loading dataset {}/{}".format(idx+1, len(files)))
        sys.stdout.flush()

        file_name = file.split("/")[-1]
        file_name = file_name.split(".")[0]
        image = load_image_L(os.path.join(dir_path, file), gamma)
        images[file_name] = image

        if save_l:
            save_image_L(image, source_path, "L", file_name)
    
    return images

def sobel_operator(image):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=image.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=image.device).unsqueeze(0).unsqueeze(0)

    edges_x = torch.nn.functional.conv2d(image, sobel_x, padding=1)
    edges_y = torch.nn.functional.conv2d(image, sobel_y, padding=1)
    edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)

    del edges_x
    del edges_y
    del sobel_x
    del sobel_y
    del image

    return edges

def perception_oriented_enhancement(edge, thred_e):
    enhanced_edge = (edge > thred_e).to(torch.float32)
    return enhanced_edge

def perception_oriented_smoothing(edge, thred_s):
    pooled_edge = torch.nn.functional.avg_pool2d(edge, kernel_size=POOLING_KERNAL_SIZE)
    smoothed_edge = torch.nn.functional.interpolate(pooled_edge, size=edge.shape[2:], mode='nearest')
    smoothed_edge = (smoothed_edge > thred_s).to(torch.float32)

    return smoothed_edge

def perceptual_sensitivity_extraction(image, thred_e, thred_s):
    edge = sobel_operator(image)
    enhanced_edge = perception_oriented_enhancement(edge, thred_e)
    smoothed_edge = perception_oriented_smoothing(enhanced_edge, thred_s)

    return smoothed_edge

def generate_sensitivity_maps(source_path):
    images = load_images(source_path, save_l=False)
    file_names = list(images.keys())

    for idx, file_name in enumerate(file_names):
        torch.cuda.empty_cache()

        sys.stdout.write('\r')
        sys.stdout.write("Generating seneitivity map {}/{}".format(idx+1, len(file_names)))
        sys.stdout.flush()
        
        sensitivity_map = perceptual_sensitivity_extraction(images[file_name], THRED_E, THRED_S)
        save_image_L(sensitivity_map, source_path, "sensitivity_maps", file_name)
        del sensitivity_map

    images = {}
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    for dataset in DATASETS:
        for scene in SCENES[dataset]:
            print(f"{dataset}/{scene}")
            torch.cuda.empty_cache()
            source_path = f"{DATASET_PATH}/{dataset}/{scene}"
            generate_sensitivity_maps(source_path)
            torch.cuda.empty_cache()
            print("\n")