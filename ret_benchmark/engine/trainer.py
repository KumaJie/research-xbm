# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import datetime
import time

import numpy as np
import torch
import faiss

from ret_benchmark.data.evaluations.eval import AccuracyCalculator
from ret_benchmark.data.evaluations.ret_metric import RetMetric
from ret_benchmark.utils.feat_extractor import feat_extractor
from ret_benchmark.utils.metric_logger import MetricLogger
from ret_benchmark.utils.log_info import log_info
from ret_benchmark.modeling.xbm import XBM


def flush_log(writer, iteration):
    for k, v in log_info.items():
        if isinstance(v, np.ndarray):
            writer.add_histogram(k, v, iteration)
        else:
            writer.add_scalar(k, v, iteration)
    for k in list(log_info.keys()):
        del log_info[k]


def do_train(
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
):
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = cfg.SOLVER.MAX_ITERS

    best_iteration = -1
    best_mapr = 0

    start_training_time = time.time()
    end = time.time()

    if cfg.XBM.ENABLE:
        logger.info(">>> use XBM")
        xbm = XBM(cfg)

    iteration = 0

    _train_loader = iter(train_loader)
    while iteration <= max_iter:
        try:
            images, targets, indices = next(_train_loader)
        except StopIteration:
            _train_loader = iter(train_loader)
            images, targets, indices = next(_train_loader)

        if (
            iteration % cfg.VALIDATION.VERBOSE == 0 or iteration == max_iter
        ) and iteration > 0:
            model.eval()
            logger.info("Validation")
            # 获取测试集的标签
            labels = val_loader[0].dataset.label_list
            labels = np.array(labels)
            # 提取测试集的所有特征
            feats = feat_extractor(model, val_loader[0], logger=logger)
            ret_metric = AccuracyCalculator(include=("precision_at_1", "mean_average_precision_at_r", "r_precision"), exclude=())
            # 计算 R@1 R@10 R@100
            # ret_metric = AccuracyCalculator(include=("recall_at_1", "recall_at_10", "recall_at_100", "mean_average_precision_at_r"), exclude=())

            ret_metric = ret_metric.get_accuracy(feats, feats, labels, labels, True)
            mapr_curr = ret_metric['mean_average_precision_at_r']
            for k, v in ret_metric.items():
                log_info[f"e_{k}"] = v

            scheduler.step(log_info[f"e_precision_at_1"])
            log_info["lr"] = optimizer.param_groups[0]["lr"]
            if mapr_curr > best_mapr:
                best_mapr = mapr_curr
                best_iteration = iteration
                logger.info(f"Best iteration {iteration}: {ret_metric}")
            else:
                logger.info(f"Performance at iteration {iteration:06d}: {ret_metric}")
            flush_log(writer, iteration)

        model.train()

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device)
        targets = targets.to(device)
        feats = model(images)

        if cfg.XBM.ENABLE and iteration > cfg.XBM.START_ITERATION:
            xbm.enqueue_dequeue(feats.detach(), targets.detach())

        loss = criterion(feats, targets, feats, targets)
        log_info["batch_loss"] = loss.item()

        if cfg.XBM.ENABLE and iteration > cfg.XBM.START_ITERATION:
            xbm_feats, xbm_targets = xbm.get()
            xbm_loss = criterion(feats, targets, xbm_feats, xbm_targets)
            log_info["xbm_loss"] = xbm_loss.item()
            loss = loss + cfg.XBM.WEIGHT * xbm_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time, loss=loss.item())
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.1f} GB",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0,
                )
            )

            log_info["loss"] = loss.item()
            flush_log(writer, iteration)

        if iteration % checkpoint_period == 0 and cfg.SAVE:
            checkpointer.save("model_{:06d}".format(iteration))
            pass

        del feats
        del loss

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

    logger.info(f"Best iteration: {best_iteration :06d} | best MAP@R {best_mapr} ")
    writer.close()

def do_test(
    cfg,
    model,
    val_loader,
    logger,
):
    logger.info("Start testing")
    model.eval()
    logger.info("Validation")
    # 获取测试集的标签
    labels = val_loader[0].dataset.label_list
    labels = np.array(labels)
    # 提取测试集的所有特征
    feats = feat_extractor(model, val_loader[0], logger=logger)
    # ret_metric = AccuracyCalculator(include=("precision_at_1", "mean_average_precision_at_r", "r_precision"), exclude=())
    # # 计算 R@1 R@10 R@100
    # ret_metric = AccuracyCalculator(include=("recall_at_1", "recall_at_10", "recall_at_100",  "recall_at_1000", "mean_average_precision_at_r"), exclude=())
    # ret_metric = ret_metric.get_accuracy(feats, feats, labels, labels, True)
    # logger.info(f"Performance : {ret_metric}")
    ret = RetMetric(feats, labels)
    recall = {}
    for k in (1, 10, 100, 1000):
        recall[f'Recall@{k}'] = ret.recall_k(k)
    logger.info(f'{cfg.MODEL.BACKBONE.NAME} Mertic: {recall}')

def do_own(
    cfg,
    model,
    val_loader,
    logger,
):
    logger.info(f"start mertic {cfg.DNAME}")
    # 获取测试集的标签
    labels = val_loader[0].dataset.label_list
    labels = np.array(labels)
    # 提取测试集的所有特征
    feats = feat_extractor(model, val_loader[0], logger=logger)
    dists, indices = get_knn(feats, feats, 1000, True)

    index_dict = val_loader[0].dataset.label_index_dict
    query = range(len(labels))
    if cfg.DNAME == 'holidays':
        query = val_loader[0].dataset.query_list

    sum_ap = 0
    # ns = 0
    for i in query:
        nres = len(index_dict[labels[i]])
        ranks = np.where(np.isin(indices[i], index_dict[labels[i]]))[0]
        # if (len(ranks) != (nres - 1)):
        #     logger.error(f"{val_loader[0].dataset.path_list[i]} not match")
        # ns += np.sum(ranks <= 2) + 1
        sum_ap += ap(ranks, nres - 1)
    logger.info(f"mAP : %.5f"%(sum_ap/len(query)))

    # logger.info(f"mAP : %.5f"%(sum_ap/len(val_loader[0].dataset.query_list)))
    # logger.info(f"ns : %.5f"%(ns/len(labels)))


def ap(ranks, nres):
    ap=0.0
    # All have an x-size of:
    recall_step=1.0/nres
        
    for ntp,rank in enumerate(ranks):
        
        # y-size on left side of trapezoid:
        # ntp = nb of true positives so far
        # rank = nb of retrieved items so far
        if rank==0: precision_0=1.0
        else:       precision_0=ntp/float(rank)

        # y-size on right side of trapezoid:
        # ntp and rank are increased by one
        precision_1=(ntp+1)/float(rank+1)
        
        ap+=(precision_1+precision_0)*recall_step/2.0
    return ap

def get_knn(
    reference_embeddings, test_embeddings, k, same
):

    d = reference_embeddings.shape[1]
    
    index = faiss.IndexFlatIP(d)
    if faiss.get_num_gpus() > 0:
        index = faiss.index_cpu_to_all_gpus(index)
    index.add(reference_embeddings)
    dists, indices = index.search(test_embeddings, k + 1)
    if same:
        return dists[:, 1:], indices[:, 1:]
    return dists[:, :k], indices[:, :k]        