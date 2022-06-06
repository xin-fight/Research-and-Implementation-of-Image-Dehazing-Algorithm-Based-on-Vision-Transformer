import os
import torch

######## Ablation Study ########
is_relative_position_bias = True  # 是否使用相对位置偏执

class Options():
    """docstring for Options"""

    def __init__(self):
        pass

    def init(self, parser):
        # update
        parser.add_argument('--is_ab', type=bool, default=False)  # 是否使用n a对比损失
        parser.add_argument('--w_loss_vgg7', type=float, default=1)  # 对比损失使用的权重
        parser.add_argument('--w_loss_CharbonnierLoss', type=float, default=1)  # CharbonnierLoss所占权重

        # global settings
        parser.add_argument('--batch_size', type=int, default=32, help='batch size')  # 设置BC
        parser.add_argument('--nepoch', type=int, default=250, help='training epochs')
        parser.add_argument('--train_workers', type=int, default=12, help='train_dataloader workers')
        parser.add_argument('--eval_workers', type=int, default=8, help='eval_dataloader workers')
        # parser.add_argument('--dataset', type=str, default='NH-HAZE')
        parser.add_argument('--dataset', type=str, default='Dense-HAZE')
        parser.add_argument('--pretrain_weights', type=str,
                            default='/media/dell/fd6f6662-7e38-4427-80c6-0d4fb1f0e8b9/work_file/2022毕业设计/Uformer_ProbSparse/log/UformerMy_Infor_NoBias/models/model_latest_epoch.pth',
                            # default='/media/dell/fd6f6662-7e38-4427-80c6-0d4fb1f0e8b9/work_file/2022毕业设计/Uformer_ProbSparse/My_best_model/Ou/U_P_ Best_PSNR: 23.3993 | the_SIMM: 0.8114.pth',
                            # default='/media/dell/fd6f6662-7e38-4427-80c6-0d4fb1f0e8b9/work_file/2022毕业设计/Uformer_ProbSparse/log/UformerMy_Infor_Ou/models/model_latest_epoch.pth',
                            help='path of pretrained_weights')
        parser.add_argument('--optimizer', type=str, default='adamw', help='optimizer for training')
        parser.add_argument('--lr_initial', type=float, default=0.0002, help='initial learning rate')
        parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')
        parser.add_argument('--gpu', type=str, default='0,1', help='GPUs')  # 使用的gpu
        parser.add_argument('--arch', type=str, default='Uformer', help='archtechture')  # 可选的结构
        parser.add_argument('--mode', type=str, default='denoising', help='image restoration mode')

        # args for saving 
        parser.add_argument('--save_dir', type=str, default='/home/ma-user/work/deNoTr/log', help='save dir')
        parser.add_argument('--save_images', action='store_true', default=False)
        parser.add_argument('--env', type=str, default='_', help='env')  # env
        parser.add_argument('--checkpoint', type=int, default=50, help='checkpoint')

        # args for Uformer
        parser.add_argument('--norm_layer', type=str, default='nn.LayerNorm', help='normalize layer in transformer')
        parser.add_argument('--embed_dim', type=int, default=32, help='dim of emdeding features')  # embeding features维度
        parser.add_argument('--win_size', type=int, default=8, help='window size of self-attention')
        parser.add_argument('--token_projection', type=str, default='linear',
                            help='linear/convoptimizer token projection')
        parser.add_argument('--token_mlp', type=str, default='leff', help='ffn/leff token mlp')
        parser.add_argument('--att_se', action='store_true', default=False, help='se after sa')

        # args for vit
        parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
        parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
        parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
        parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
        parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
        parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
        parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
        parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')

        # args for training
        parser.add_argument('--train_ps', type=int, default=128, help='patch size of training sample')  # 训练样本的补丁大小
        parser.add_argument('--resume', action='store_true', default=True)
        parser.add_argument('--train_dir', type=str, default='../datasets/SIDD/train', help='dir of train data')  # 训练数据
        parser.add_argument('--val_dir', type=str, default='../datasets/SIDD/val',
                            help='dir of train data')  # dir of train data
        # action='store_true'，只要运行时该变量有传参就将该变量设为True。
        parser.add_argument('--warmup', action='store_true', default=False, help='warmup')  # warmup是一种学习率优化方法
        parser.add_argument('--warmup_epochs', type=int, default=3, help='epochs for warmup')

        return parser
