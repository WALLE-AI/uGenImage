"""T3：感知损失与对抗损失（对应 OPTIMIZATION_PLAN.md 阶段 T3）。

诊断显示重建误差的 56%~60% 集中在高频。L1 优化的是逐像素条件中位数，
在信息不足时的最优解就是模糊的平均图像 —— 纹理和边缘从原理上就不会被还原。
锐度只能来自感知损失 + 对抗损失。

感知骨干两选一，都用本机已缓存的 ImageNet 权重，训练时全程冻结：
  vgg16    LPIPS 的标准骨干（~/.cache/torch/hub/checkpoints/vgg16-397923af.pth）
  resnet50 备选，无需任何下载
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# ImageNet 归一化常数；输入按本项目约定是 [-1,1]
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class PerceptualLoss(nn.Module):
    """多层特征匹配。各层特征先做通道维 L2 归一化再比较，

    避免深层的大激活值主导损失（这也是 LPIPS 的做法）。
    """

    def __init__(self, backbone='vgg16'):
        super().__init__()
        if backbone == 'vgg16':
            net = torchvision.models.vgg16(weights='IMAGENET1K_V1').features
            self.slices = nn.ModuleList([
                net[:4], net[4:9], net[9:16], net[16:23], net[23:30],
            ])
        elif backbone == 'resnet50':
            net = torchvision.models.resnet50(weights='IMAGENET1K_V1')
            stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
            self.slices = nn.ModuleList([stem, net.layer1, net.layer2, net.layer3])
        else:
            raise ValueError(f"未知感知骨干: {backbone}")

        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer('mean', _MEAN)
        self.register_buffer('std', _STD)

    def train(self, mode=True):
        return super().train(False)   # 感知网络永远保持 eval

    def _prep(self, x):
        return ((x + 1) / 2 - self.mean) / self.std

    def forward(self, pred, target):
        h_p, h_t = self._prep(pred), self._prep(target)
        loss = 0.0
        for s in self.slices:
            h_p, h_t = s(h_p), s(h_t)
            loss = loss + F.l1_loss(F.normalize(h_p, dim=1),
                                    F.normalize(h_t, dim=1))
        return loss / len(self.slices)


class PatchDiscriminator(nn.Module):
    """PatchGAN，感受野约 70×70。判别的是局部真实性，正是纹理所在的尺度。"""

    def __init__(self, in_ch=3, ch=64, n_layers=3):
        super().__init__()
        sn = nn.utils.spectral_norm
        layers = [nn.Conv2d(in_ch, ch, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        mult = 1
        for i in range(1, n_layers + 1):
            prev, mult = mult, min(2 ** i, 8)
            stride = 2 if i < n_layers else 1
            layers += [sn(nn.Conv2d(ch * prev, ch * mult, 4, stride, 1, bias=False)),
                       nn.BatchNorm2d(ch * mult), nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(ch * mult, 1, 4, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def hinge_d_loss(real, fake):
    return 0.5 * (F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean())


def adaptive_weight(rec_loss, adv_loss, last_layer, max_weight=1e4):
    """VQGAN 原论文的自适应对抗权重：让两个损失在最后一层上的梯度量级相当。

    避免手调一个对数据集敏感的固定权重。
    """
    g_rec = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
    g_adv = torch.autograd.grad(adv_loss, last_layer, retain_graph=True)[0]
    w = g_rec.norm() / (g_adv.norm() + 1e-4)
    return w.clamp(0.0, max_weight).detach()
