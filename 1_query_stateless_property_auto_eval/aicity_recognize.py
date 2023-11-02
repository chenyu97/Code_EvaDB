import json
import os
import argparse
import torch
import torch.multiprocessing
from tqdm import tqdm
from configs import get_default_config
#from models.backbones.CLIP import clip
#from models.backbones import open_clip
import torch.nn.functional as F
from utils.utils import set_seed
from collections import OrderedDict
from models import build_model
#from datasets import build_dataloader
import numpy as np
from PIL import Image
import torchvision
import time
import cv2


def build_transform(is_train, cfg):
    if is_train:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((cfg.DATA.SIZE,cfg.DATA.SIZE)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(cfg.DATA.SIZE, scale=(1, 1.)),
            torchvision.transforms.RandomApply([torchvision.transforms.RandomRotation(0)],p=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    return transform

def prepare_recognize(property_name):
    if property_name not in ["color", "type", "direction"]:
        raise ValueError("CLIP does not support recognition for property: " + property_name)

    cfg_name = "CLIP_recognition_" + property_name
    cfg = get_default_config()
    cfg.merge_from_file(os.path.join(cfg.DATA.ROOT_DIR, 'configs', cfg_name + ".yaml"))

    set_seed(cfg.TRAIN.SEED)
    use_cuda = True

    transform = build_transform(is_train=False, cfg=cfg)

    model = build_model(cfg)
    CKPT_SAVE_DIR = os.path.join(cfg.DATA.DATA_DIR, cfg.DATA.CHECKPOINT_PATH, cfg_name)
    LOG_SAVE_DIR = os.path.join(cfg.DATA.DATA_DIR, cfg.DATA.LOG_PATH)
    os.makedirs(CKPT_SAVE_DIR,exist_ok = True)
    os.makedirs(LOG_SAVE_DIR,exist_ok = True)

    INFERENCE_DIR = os.path.join(cfg.DATA.DATA_DIR, cfg.TEST.INFERENCE_FROM)

    #print("Inference model from weight %s"%INFERENCE_DIR)
        
    checkpoint = torch.load(INFERENCE_DIR)
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[7:] # remove `module.`
        # print(name)
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)

    if use_cuda:
        model.cuda()
    model.eval()

    return transform, model

def recognize(transform, model, img, reference):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_PIL = Image.fromarray(img_rgb) # Adapt to CLIP_recognize model
    img_tensor = transform(img_PIL)
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        cls_logit = model(img_tensor.cuda(), img_tensor.cuda())  
        predict = cls_logit.argmax(1)
        return reference[int(predict[0].data.cpu().numpy())]

def recognize_id(transform, model, img, reference):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_PIL = Image.fromarray(img_rgb) # Adapt to CLIP_recognize model
    img_tensor = transform(img_PIL)
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        cls_logit = model(img_tensor.cuda(), img_tensor.cuda())  
        predict = cls_logit.argmax(1)
        return int(predict[0].data.cpu().numpy())

