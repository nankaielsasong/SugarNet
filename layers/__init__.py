# encoding: utf-8

import torch.nn.functional as F

from .triplet_loss import TripletLoss


def make_loss(cfg):
    sampler = cfg.DATALOADER.SAMPLER
    # triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
    triplet = TripletLoss(0.3)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif cfg.DATALOADER.SAMPLER == 'triplet':
        def loss_func(score, feat, target, e):
            return triplet(feat, target, e)[0]
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, e):
            return F.cross_entropy(score, target) + triplet(feat, target, e)[0]
    elif cfg.DATALOADER.SAMPLER == 'multiple_softmax_triplet':
        def loss_func(feat1, score, feat, target, alpha, epoch_num):
            triplet_losses, dis_aps, dis_ans = triplet(feat[0], target, epoch_num)
            for i, f in enumerate(feat):
                if i == 0:
                    continue
                triplet_loss, dis_ap, dis_an = triplet(f, target, epoch_num)
                triplet_losses += triplet_loss
                dis_aps += dis_ap
                dis_ans += dis_an
            softmax_losses = [F.cross_entropy(output, target) for output in score]
            loss = sum(softmax_losses) + triplet_losses#  + pow(alpha-1, 2)
            return loss, dis_aps, dis_ans, triplet_losses
    else:
        print('expected sampler should be softmax, triplet or softmax_triplet, '
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func


