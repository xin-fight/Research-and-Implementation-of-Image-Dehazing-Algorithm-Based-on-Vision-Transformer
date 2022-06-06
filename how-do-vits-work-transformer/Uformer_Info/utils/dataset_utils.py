import torch
import os


### rotate and flip
class Augment_RGB_torch:
    def __init__(self):
        pass

    def transform0(self, torch_tensor):
        return torch_tensor

    def transform1(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=1, dims=[-1, -2])
        return torch_tensor

    def transform2(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=2, dims=[-1, -2])
        return torch_tensor

    def transform3(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=3, dims=[-1, -2])
        return torch_tensor

    def transform4(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor

    def transform5(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=1, dims=[-1, -2])).flip(-2)
        return torch_tensor

    def transform6(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=2, dims=[-1, -2])).flip(-2)
        return torch_tensor

    def transform7(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=3, dims=[-1, -2])).flip(-2)
        return torch_tensor


### mix two images
"""混合样本数据增强（Mixed Sample Data Augmentation）:
【深度学习】Mixup: Beyond Empirical Risk Minimization: https://blog.csdn.net/zzc15806/article/details/80696787
    Mixup算法的核心思想是按一定的比例随机混合两个训练样本及其标签
    这种混合方式不仅能够增加样本的多样性，并且能够使不同类别的决策边界过渡更加平滑
    减少了一些难例样本的误识别，模型的鲁棒性得到提升，训练时也比较稳定，减少了一些难例样本的误识别，模型的鲁棒性得到提升，训练时也比较稳定
"""


class MixUp_AUG:
    def __init__(self):
        # distributions 包含可参数化的概率分布和采样函数
        # 参考网址：https://blog.csdn.net/qq_40206371/article/details/122230739
        # 贝塔分布
        self.dist = torch.distributions.beta.Beta(torch.tensor([1.2]), torch.tensor([1.2]))

    # target, input_  [B, C, W, H]
    def aug(self, rgb_gt, rgb_noisy):
        bs = rgb_gt.size(0)
        indices = torch.randperm(bs)  # 返回一个从0到n-1的随机整数排列
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]

        # .rsample(sample_shape) 如果分布参数是批处理的，则生成一个 sample_shape 形状的重新参数化样本或 sample_shape 形状的重新参数化样本批次。
        lam = self.dist.rsample((bs, 1)).view(-1, 1, 1, 1).cuda()

        rgb_gt = lam * rgb_gt + (1 - lam) * rgb_gt2
        rgb_noisy = lam * rgb_noisy + (1 - lam) * rgb_noisy2

        return rgb_gt, rgb_noisy
