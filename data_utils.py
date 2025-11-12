import os
import clip
import torch
import pickle
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from pytorchcv.model_provider import get_model as ptcv_get_model

LABEL_FILES_ROOT = "./data/classes_name/"
DATASET_ROOTS = "./dataset/"


def get_data(dataset_name, split="train", transform=None):

    root = os.path.join(DATASET_ROOTS, dataset_name, f"{dataset_name}_{split}")
    return ImageFolder(root=root, transform=transform)

def get_targets_only(dataset_name, split="train"):
    
    dataset = get_data(dataset_name, split)
    return dataset.targets

def get_class_names(dataset_name):

    classes_file = os.path.join(LABEL_FILES_ROOT, dataset_name+"_classes.txt")
    with open(classes_file, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    return classes

def get_resnet_imagenet_preprocess():
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=target_mean, std=target_std)
    ])
    return preprocess

def get_target_model(target_name, device):
    if target_name.startswith("clip_"):
        target_name = target_name[5:]
        model, preprocess = clip.load(target_name, device=device)
        target_model = lambda x: model.encode_image(x).float()
    
    elif target_name == 'resnet18_places': 
        target_model = models.resnet18(pretrained=False, num_classes=365).to(device)
        state_dict = torch.load('data/resnet18_places365.pth.tar')['state_dict']
        new_state_dict = {}
        for key in state_dict:
            if key.startswith('module.'):
                new_state_dict[key[7:]] = state_dict[key]
        target_model.load_state_dict(new_state_dict)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
        
    elif target_name == 'resnet18_cub':
        target_model = ptcv_get_model("resnet18_cub", pretrained=True).to(device)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
    
    elif target_name.endswith("_v2"):
        target_name = target_name[:-3]
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V2".format(target_name_cap))
        target_model = eval("models.{}(weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()
        
    else:
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()
    
    return target_model, preprocess