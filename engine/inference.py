# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import logging

import torch
from ignite.engine import Engine
import os
from utils.reid_metric import R1_mAP


def create_supervised_evaluator(model, metrics,
                                device=None, num_gpus=1):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """

    if num_gpus > 1:
        device_id = [0, 1]#list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(',')))
        model = torch.nn.DataParallel(model, device_ids=device_id)

    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.cuda()
            feat = model(data)
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def inference(
        cfg,
        model,
        val_loader,
        num_query,
        num_gpus
):
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Start inferencing")
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, cfg.DATASETS.NAMES)},
                                            device=device, num_gpus=num_gpus)

    evaluator.run(val_loader)
    cmc, mAP = evaluator.state.metrics['r1_mAP']
    logger.info('Validation Results')
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
