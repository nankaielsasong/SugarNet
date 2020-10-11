# encoding: utf-8

import argparse
import os
import sys
import torch

from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.trainer import do_train
from modeling import build_model
from layers import make_loss
from solver import make_optimizer, WarmupMultiStepLR

from utils.logger import setup_logger
from utils.visualize import visualize_model

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def train(cfg, num_gpus=1, load_LRpart1_Path=None, load_LRpart1_Optimizer_Path=None):
    # prepare dataset
    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)
    # prepare model
    model = build_model(cfg, num_classes)

    if load_LRpart1_Path != None:
        # model_dict = model.state_dict()
        # res_pretrained_dict = torch.load(load_LRpart1_Path)
        # res_pretrained_dict = {k: v for k, v in res_pretrained_dict.items() if k in model_dict}
        # model_dict.update(res_pretrained_dict)
        # model.load_state_dict(model_dict)
        model.load_state_dict(torch.load(load_LRpart1_Path))
        print("finished loading LRpart1 weights")


    optimizer = make_optimizer(cfg, model)
    if load_LRpart1_Optimizer_Path != None:
        # model_dict = model.state_dict()
        # res_pretrained_dict = torch.load(load_LRpart1_Path)
        # res_pretrained_dict = {k: v for k, v in res_pretrained_dict.items() if k in model_dict}
        # model_dict.update(res_pretrained_dict)
        # model.load_state_dict(model_dict)
        optimizer.load_state_dict(torch.load(load_LRpart1_Optimizer_Path))
        print("finished loading LRpart1_Optimizer weights")
        optimizer = optimizer.cpu()
    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                  cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

    loss_func = make_loss(cfg)

    arguments = {}

    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_func,
        num_query,
	num_gpus
    )


def visualize(cfg):
    print('start visualizing......')
    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)
    model = build_model(cfg, num_classes)
    # model = model.to(cfg.MODEL.DEVICE)
    visualize_model(model, 'base_resnet')
    

def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/softmax_triplet.yml", help="path to config file", type=str
    )
    parser.add_argument(
        "--load_LRpart1_Path", default=None, help="path to config file", type=str
    )
    parser.add_argument(
        "--load_LRpart1_Optimizer_Path", default=None, help="Optimizer", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    # num_gpus = 3
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    if args.load_LRpart1_Path != None:
        print("the traing will load retrained weights from " + args.load_LRpart1_Path)
    logger.info("Running with config:\n{}".format(cfg))

    cudnn.benchmark = True
    
    # visualize(cfg)
    train(cfg, num_gpus, args.load_LRpart1_Path, args.load_LRpart1_Optimizer_Path)


if __name__ == '__main__':
    main()
