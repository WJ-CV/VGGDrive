import cv2
from PIL import Image
import numpy as np
import os
import pickle
import json
import ast
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
from vggt.utils.load_fn import load_and_preprocess_images
import torch.nn.functional as F


def fetch_img_list(example):
    process_img_list = []
    depth_map_list = []
    content_list = example["messages"][0]["content"]
    for content in content_list:
        if 'image' not in content:
            continue
        image_path = content['image']
        resized_height, resized_width = content["resized_height"], content["resized_width"]
        img_path = image_path.replace("file://", "")
        process_img_list.append(img_path)
    process_imgs = load_and_preprocess_images(process_img_list)

    return process_imgs 

def fetch_img_list_navsim(example):
    process_img_list = []
    for exp in example["messages"]:
        if exp.get("role") == "user":
            content_list = exp.get("content", [])
    for content in content_list:
        if 'image' not in content:
            continue
        image_path = content['image']
        resized_height, resized_width = content["resized_height"], content["resized_width"]
        img_path = image_path.replace("file://", "")
        process_img_list.append(img_path)

    process_imgs = load_and_preprocess_images(process_img_list)

    return process_imgs


def lidar2img(example, resized_H, resized_W, img_H, img_W): # 588, 1036   1080, 1920
    img2lidar = []
    intrinsic_ = np.array(example['intrinsics'])
    sensor2lidar_rotation_ = np.array(example['sensor2lidar_rotation'])
    sensor2lidar_translation_ = np.array(example['sensor2lidar_translation'])

    # 计算缩放因子
    scale_x = resized_W / img_W
    scale_y = resized_H / img_H

    for i in range(len(intrinsic_)):    # 缩放内参
        intrinsic = intrinsic_[i]
        # 缩放 fx, fy, cx, cy
        intrinsic[0][0] *= scale_x   # fx
        intrinsic[0][2] *= scale_x   # cx
        intrinsic[1][1] *= scale_y   # fy
        intrinsic[1][2] *= scale_y   # cy
        intrinsic_[i] = intrinsic  

    for i in range(len(intrinsic_)):
        intrinsic = intrinsic_[i]
        sensor2lidar_rotation = sensor2lidar_rotation_[i]
        sensor2lidar_translation = sensor2lidar_translation_[i]

        viewpad = np.eye(4)
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

        lidar2cam_r_ = np.linalg.inv(sensor2lidar_rotation)
        lidar2cam_t_ = sensor2lidar_translation @ lidar2cam_r_.T
        lidar2cam_rt_ = np.eye(4)
        lidar2cam_rt_[:3, :3] = lidar2cam_r_.T
        lidar2cam_rt_[3, :3] = -lidar2cam_t_
        lidar2cam_rt = (viewpad @ lidar2cam_rt_.T)
        img2lidar.append(np.linalg.inv(lidar2cam_rt))

    return img2lidar




