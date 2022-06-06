import copy
import re

import numpy as np
import torch

import ops.norm as norm
import ops.My_tests as tests


def normalize_filter(bs, ws):
    bs = {k: v.float() for k, v in bs.items()}
    ws = {k: v.float() for k, v in ws.items()}

    norm_bs = {}
    for k in bs:
        ws_norm = torch.norm(ws[k], dim=0, keepdim=True)
        bs_norm = torch.norm(bs[k], dim=0, keepdim=True)
        norm_bs[k] = ws_norm / (bs_norm + 1e-7) * bs[k]

    return norm_bs


def ignore_bn(ws):
    ignored_ws = {}
    for k in ws:
        if len(ws[k].size()) < 2:
            ignored_ws[k] = torch.zeros(size=ws[k].size(), device=ws[k].device)
        else:
            ignored_ws[k] = ws[k]
    return ignored_ws


def ignore_running_stats(ws):
    return ignore_kw(ws, ["num_batches_tracked"])


def ignore_kw(ws, kws=None):
    kws = [] if kws is None else kws

    ignored_ws = {}
    for k in ws:
        if any([re.search(kw, k) for kw in kws]):
            ignored_ws[k] = torch.zeros(size=ws[k].size(), device=ws[k].device)
        else:
            ignored_ws[k] = ws[k]
    return ignored_ws


def rand_basis(ws, gpu=True):
    return {k: torch.randn(size=v.shape, device="cuda" if gpu else None) for k, v in ws.items()}


def create_bases(model, kws=None, gpu=True):
    """
    :param model: 加载预训练权重的模型
    :param kws:
    :param gpu:
    :return:
    """
    kws = [] if kws is None else kws
    ws0 = copy.deepcopy(model.state_dict())
    # 随机初始化两个和ws0内容同样大小的bases [{}, {}]
    bases = [rand_basis(ws0, gpu) for _ in range(2)]  # Use two bases
    # bases [{ws_norm / (bs_norm + 1e-7) * bs[k]}, {}]
    bases = [normalize_filter(bs, ws0) for bs in bases]
    # 设len(bs[k].size()) < 2为0，其他不变
    bases = [ignore_bn(bs) for bs in bases]
    # 在kws中的key的值被置为0
    bases = [ignore_kw(bs, kws) for bs in bases]

    return bases


def get_loss_landscape(model, n_ff, dataset, transform=None,
                       bases=None, kws=None,
                       cutoffs=(0.0, 0.9), bins=np.linspace(0.0, 1.0, 11), verbose=False, period=10, gpu=True,
                       x_min=-1.0, x_max=1.0, n_x=11, y_min=-1.0, y_max=1.0, n_y=11):
    """
    lls.get_loss_landscape(
        model, 1, dataset_train, transform=transform,  # transform = mixup_function(train_args) MixUp
        kws=["pos_embed", "relative_position"],
        x_min=-1.0 * scale, x_max=1.0 * scale, n_x=n, y_min=-1.0 * scale, y_max=1.0 * scale, n_y=n, gpu=gpu,)
    """
    model = model.cuda() if gpu else model.cpu()
    model = copy.deepcopy(model)
    ws0 = copy.deepcopy(model.state_dict())
    kws = [] if kws is None else kws
    # bases [{}, {}] 经过norm，len(bs[k].size()) < 2为0，kws中值为0
    bases = create_bases(model, kws, gpu) if bases is None else bases

    # np.linspace(-1，1，11)  有11个值
    xs = np.linspace(x_min, x_max, n_x)
    ys = np.linspace(y_min, y_max, n_y)
    # np.meshgrid返回list,有两个元素,第一个元素是X轴的取值,第二个元素是Y轴的取值
    ratio_grid = np.stack(np.meshgrid(xs, ys), axis=0).transpose((1, 2, 0))  # (2, 11, 11)->(11, 11, 2)

    # 对画图的 每一个点 进行计算
    metrics_grid = {}
    for ratio in ratio_grid.reshape([-1, 2]):  # (121, 2)
        # 修改模型加载的参数
        ws = copy.deepcopy(ws0)

        ### 让模型加载修改后的权重 ###
        # ratio (2,): [-1.            -1.] 相当于画网格的坐标
        # bases       [{与ws0同样大小}, {}] 经过norm，len(bs[k].size()) < 2为0，kws中值为0
        ### 让bases中第i个{}的所有值 与 ratio对应第i项相乘 ###
        gs = [{k: r * bs[k] for k in bs} for r, bs in zip(ratio, bases)]
        # {k:bases中 k对应两个{}中值相加 + ws[k]}
        gs = {k: torch.sum(torch.stack([g[k] for g in gs]), dim=0) + ws[k] for k in gs[0]}
        model.load_state_dict(gs)

        # 使用修改后的参数得到各种指标
        print("Grid: ", ratio, end=", ")
        *metrics, cal_diag = tests.test(model, n_ff, dataset, transform=transform,
                                        cutoffs=cutoffs, bins=bins, verbose=verbose, period=period, gpu=gpu)

        # l1_norm += torch.norm(param, 1) 对模型参数进行norm
        l1, l2 = norm.l1(model, gpu).item(), norm.l2(model, gpu).item()
        metrics_grid[tuple(ratio)] = (l1, l2, *metrics)

        torch.cuda.empty_cache()

    return metrics_grid
