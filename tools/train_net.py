# encoding: utf-8

# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import sys
sys.path.append('..')

import argparse
import torch
import torch.nn.functional as F
from torch import nn
import torchvision.models as models
import os

from ret_benchmark.config import cfg
from ret_benchmark.data import build_data
from ret_benchmark.engine.trainer import do_train
from ret_benchmark.losses import build_loss
from ret_benchmark.modeling import build_model
from ret_benchmark.solver import build_lr_scheduler, build_optimizer
from ret_benchmark.utils.logger import setup_logger
from ret_benchmark.utils.checkpoint import Checkpointer
from tensorboardX import SummaryWriter

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

class CustomResNet50(torch.nn.Module):
    def __init__(self, num_classes=256):
        super(CustomResNet50, self).__init__()
        # 加载预训练的 ResNet50 模型
        self.resnet50 = models.resnet50(weights='IMAGENET1K_V1')

        # 修改全连接层，将输出维度改为256
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)
        torch.nn.init.kaiming_normal_(self.resnet50.fc.weight, a=0, mode="fan_out")
        torch.nn.init.constant_(self.resnet50.fc.bias, 0.0)

    def forward(self, x):
        x = self.resnet50(x)
        x = F.normalize(x)
        return x




def train(cfg):
    logger = setup_logger(name="Train", level=cfg.LOGGER.LEVEL)
    logger.info(cfg)
    # model = build_model(cfg)
    model = CustomResNet50()
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    # if len(os.environ["CUDA_VISIBLE_DEVICES"]) > 1:
    #     model = torch.nn.DataParallel(model)

    criterion = build_loss(cfg)

    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    train_loader = build_data(cfg, is_train=True)
    val_loader = build_data(cfg, is_train=False)

    logger.info(train_loader.dataset)
    for x in val_loader:
        logger.info(x.dataset)

    arguments = dict()
    arguments["iteration"] = 0

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    ckp_save_path = os.path.join(cfg.SAVE_DIR, cfg.NAME)
    os.makedirs(ckp_save_path, exist_ok=True)
    checkpointer = Checkpointer(model, optimizer, scheduler, ckp_save_path)

    tb_save_path = os.path.join(cfg.TB_SAVE_DIR, cfg.NAME)
    os.makedirs(tb_save_path, exist_ok=True)
    writer = SummaryWriter(tb_save_path)

    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        checkpointer,
        writer,
        device,
        checkpoint_period,
        arguments,
        logger,
    )


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="Train a retrieval network")
    parser.add_argument(
        "--cfg", dest="cfg_file", help="config file", default=None, type=str
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg.merge_from_file(args.cfg_file)
    train(cfg)
