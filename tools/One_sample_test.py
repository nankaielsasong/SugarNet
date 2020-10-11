# encoding: utf-8

import argparse
import os
import sys
from os import mkdir
from PIL import Image
import numpy as np
import scipy.io as scio

import torch
from torch.backends import cudnn
import torchvision.transforms as T

sys.path.append('.')
from config import cfg
from modeling import build_model



def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = '../Person_result'
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    cudnn.benchmark = True

    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    transform = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        normalize_transform
    ])

    model = build_model(cfg, 751)
    model.load_state_dict(torch.load(cfg.TEST.WEIGHT))
    model.to(cfg.MODEL.DEVICE)
    model.eval()
    test_pth = '/home/dongshichao/Dsc/D_paper/reid_baseline-master/liunian/reid_baseline-master/tools/1.jpg'
    test_img = Image.open(test_pth).convert('RGB')

    test_dir_pth = '/home/dongshichao/Dsc/D_paper/vgg_baseline/Market-1501/bounding_box_train'
    test_dir = os.listdir(test_dir_pth)

    test_img = transform(test_img)
    with torch.no_grad():
        key_vector = model(test_img.expand(1, 3, 384, 192).cuda())
        key_vector = np.array(key_vector.cpu())
    mat_pth = 'market_train.mat'
    result = []
    pose_result = []
    global_result = []
    if mat_pth and not os.path.exists(mat_pth):
        dir_list = []
        for img_name in test_dir:
            img = Image.open(os.path.join(test_dir_pth, img_name)).convert('RGB')
            img = transform(img)
            dir_list.append([img_name, img.expand(1, 3, 384, 192)])

        save_mat = {}
        for i, dir in enumerate(dir_list):
            img_name, img = dir
            if i % 100 == 0:
                print(str(i)+'/'+str(len(dir_list)))
            with torch.no_grad():
                feature_vector = model(img.cuda())
                feature_vector = np.array(feature_vector.cpu())
                save_mat.update({img_name: feature_vector})
                score = np.sqrt(np.sum((feature_vector - key_vector)**2))
                result.append([score, img_name])
        scio.savemat(mat_pth, save_mat)
    else:
        dict_mat = scio.loadmat(mat_pth)
        i = 0
        for (img_name, feature_vector) in dict_mat.items():
            i += 1
            if i % 100 == 0:
                print(str(i)+'/'+str(len(dict_mat)))
            try:
                # pose_score = np.sqrt(np.sum((feature_vector[:, 4096:] - key_vector[:, 4096:]) ** 2))
                # global_score = np.sqrt(np.sum((feature_vector[:, :4096] - key_vector[:, :4096]) ** 2))
                score = np.sqrt(np.sum((feature_vector - key_vector) ** 2))
                result.append([score, img_name])
                # pose_result.append([pose_score, img_name])
                # global_result.append([global_score, img_name])
            except:
                pass

    result.sort()
    # pose_result.sort()
    # global_result.sort()

    for i, re in enumerate(result):
        if i == 20:
            break
        score, img_name = re
        img = Image.open(os.path.join(test_dir_pth, img_name))
        re_outpth = os.path.join(output_dir, 'result')
        if re_outpth and not os.path.exists(re_outpth):
            mkdir(re_outpth)
        img.save(re_outpth+'/'+str(score)+'_'+img_name)
'''
    for i, re in enumerate(pose_result):
        if i == 20:
            break
        score, img_name = re
        img = Image.open(os.path.join(test_dir_pth, img_name))
        re_outpth = os.path.join(output_dir, 'pose_result')
        if re_outpth and not os.path.exists(re_outpth):
            mkdir(re_outpth)
        img.save(re_outpth + '/' + str(score) + '_' + img_name)

    for i, re in enumerate(global_result):
        if i == 20:
            break
        score, img_name = re
        img = Image.open(os.path.join(test_dir_pth, img_name))
        re_outpth = os.path.join(output_dir, 'global_result')
        if re_outpth and not os.path.exists(re_outpth):
            mkdir(re_outpth)
        img.save(re_outpth + '/' + str(score) + '_' + img_name)
'''

if __name__ == '__main__':
    main()

