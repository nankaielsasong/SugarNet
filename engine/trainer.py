# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import os
import logging

import torch
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer, Checkpoint, DiskSaver
from ignite.metrics import RunningAverage

from utils.reid_metric import R1_mAP


epoch_tripletloss_file = None
def create_supervised_trainer(model, optimizer, loss_fn,
                              device=None, num_gpus=1, cfg=None):

    global epoch_tripletloss_file
    epoch_tripletloss_file = open(os.path.join(cfg.OUTPUT_DIR, "epoch_tripletloss.txt"), "w", encoding="utf-8")


    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if num_gpus > 1:
        id_list = []
        for i in range(num_gpus):
            id_list.append(i)
        model = torch.nn.DataParallel(model, device_ids=id_list)
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        img, target = batch
        img = img.cuda()
        target = target.cuda()
        score, feat = model(img, target)
        result = score[0]
        loss, dis_ap, dis_an, triplet_loss = loss_fn(None, score, feat, target, None, engine.state.epoch)
        loss.backward()
        optimizer.step()
        # compute acc
        acc = (result.max(1)[1] == target).float().mean()
        return loss.item(), acc.item(), dis_ap.mean().item(), dis_an.mean().item(), triplet_loss.item()

    return Engine(_update)


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
        id_list = []
        for i in range(num_gpus):
            id_list.append(i)
        model = torch.nn.DataParallel(model, device_ids=id_list)
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


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
	num_gpus
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS
    
    logger = logging.getLogger("reid_baseline.train")

    '''
    log_path = './logs/BNNeck/no_channel_split.log'
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    datefmt = '%a %d %b %Y %H:%M:%S'
    formatter = logging.Formatter(fmt, datefmt)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    '''

    logger.info("Start training")
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device, num_gpus, cfg)
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, cfg.DATASETS.NAMES)}, device=device)
    
    to_save = {'model': model, 'optimizer':optimizer}
    handler = Checkpoint(to_save, DiskSaver(output_dir, create_dir=True), n_saved=10)
    # checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=checkpoint_period), handler)
    #trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     # 'optimizer': optimizer})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')
    # RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'alpha')
    RunningAverage(output_transform=lambda x: x[3]).attach(trainer, 'dis_an')
    RunningAverage(output_transform=lambda x: x[4]).attach(trainer, 'triplet_loss')

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, "
                        "Base Lr: {:.2e}"
                        .format(engine.state.epoch, iter, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc']
                                , scheduler.get_lr()[0]))

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):

        global epoch_tripletloss_file
        epoch_tripletloss_file.write(str(engine.state.metrics['triplet_loss']) + ', ')
        epoch_tripletloss_file.flush()

        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    trainer.run(train_loader, max_epochs=epochs)


