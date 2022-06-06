import torch
import torch.nn as nn
import os
from collections import OrderedDict


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze(model):
    for p in model.parameters():
        p.requires_grad = True


def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)


def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    model_out_path = os.path.join(model_dir, "model_epoch_{}_{}.pth".format(epoch, session))
    torch.save(state, model_out_path)


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    print('load weight path:' + weights)
    try:
        # strict=False 预训练权重中与新构建网络中匹配层的键值就进行使用，没有的就默认初始化。
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k  # 去除前面的module.
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def load_checkpoint_CPU(model, weights):
    checkpoint = torch.load(weights, map_location=torch.device('cpu'))
    print('load weight path:' + weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k  # 去除前面的module.
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def load_checkpoint_multigpu(model, weights):
    checkpoint = torch.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)


def load_start_epoch(weights):
    checkpoint = torch.load(weights)
    epoch = checkpoint["epoch"]
    return epoch


def load_optim(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for p in optimizer.param_groups: lr = p['lr']
    return lr


def get_arch(opt):
    from Uformer_Info.My_model_1 import UNet, Uformer
    # from My_model import UNet, Uformer
    arch = opt.arch

    print('You choose ' + arch + '...')
    if arch == 'UNet':
        model_restoration = UNet(dim=opt.embed_dim)

    elif arch == 'Uformer':
        '''
        train_ps: 训练样本的补丁大小; embed_dim: embeding features维度; win_size: window size of self-attention = 8;
        token_projection: linear/conv token projection = linear;    token_mlp: ffn/leff token mlp = leff
        '''
        model_restoration = Uformer(img_size=opt.train_ps, embed_dim=opt.embed_dim, win_size=opt.win_size,
                                    token_projection=opt.token_projection, token_mlp=opt.token_mlp)
    elif arch == 'Uformer16':
        model_restoration = Uformer(img_size=opt.train_ps, embed_dim=16, win_size=8, token_projection='linear',
                                    token_mlp='leff')
    elif arch == 'Uformer32':
        model_restoration = Uformer(img_size=opt.train_ps, embed_dim=32, win_size=8, token_projection='linear',
                                    token_mlp='leff')
    else:
        raise Exception("Arch error!")

    return model_restoration
