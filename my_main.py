import copy
import json
import os
import warnings
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from torchvision import transforms
import torchvision
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from tqdm import trange
import random

from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler, LatentGaussianDiffusionTrainer, \
    LatentGaussianDiffusionSampler
from model import Spk_UNet
# 移除FID/IS计算相关导入

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
## 参数解析 ##
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=42, type=int, help='随机种子')
parser.add_argument('--train', action='store_true', default=False, help='从头训练')
parser.add_argument('--eval', action='store_true', default=False, help='加载模型并生成图像')
parser.add_argument('--dataset', type=str, default='cifar10', help='数据集名称')
parser.add_argument('--sample_type', type=str, default='ddpm', help='采样类型')
parser.add_argument('--wandb', action='store_true', default=False, help='使用wandb记录训练')
# 脉冲UNet参数
parser.add_argument('--ch', default=128, type=int, help='UNet基础通道数')
parser.add_argument('--ch_mult', default=[1, 2, 2, 4], help='通道倍增系数')
parser.add_argument('--attn', default=[], help='注意力机制应用的层级')
parser.add_argument('--num_res_blocks', default=2, type=int, help='每个层级的残差块数量')
parser.add_argument('--img_size', default=32, type=int, help='图像尺寸')
parser.add_argument('--dropout', default=0.1, type=float, help='残差块的dropout率')
parser.add_argument('--timestep', default=4, type=int, help='SNN时间步')
parser.add_argument('--img_ch', type=int, default=3, help='图像通道数')
# 高斯扩散参数
parser.add_argument('--beta_1', default=1e-4, type=float, help='初始beta值')
parser.add_argument('--beta_T', default=0.02, type=float, help='最终beta值')
parser.add_argument('--T', default=1000, type=int, help='总扩散步数')
parser.add_argument('--mean_type', default='epsilon', help='预测变量类型:[xprev, xstart, epsilon]')
parser.add_argument('--var_type', default='fixedlarge', help='方差类型:[fixedlarge, fixedsmall]')
# 训练参数
parser.add_argument('--resume', default=False, help="加载预训练模型")
parser.add_argument('--resume_model', type=str, help='预训练模型路径')
parser.add_argument('--lr', default=2e-4, help='目标学习率')
parser.add_argument('--grad_clip', default=1., help="梯度裁剪阈值")
parser.add_argument('--total_steps', type=int, default=500000, help='总训练步数')
parser.add_argument('--warmup', default=5000, help='学习率预热步数')
parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
parser.add_argument('--ema_decay', default=0.9999, help="EMA衰减率")
parser.add_argument('--parallel', default=True, help='多GPU训练')
# 日志和采样参数
parser.add_argument('--logdir', default='./log', help='日志目录')
parser.add_argument('--sample_size', type=int, default=64, help="采样图像数量")
parser.add_argument('--sample_step', type=int, default=5000, help='采样频率')
# 评估参数（移除FID相关参数）
parser.add_argument('--save_step', type=int, default=0, help='保存检查点频率，0为训练时禁用')
parser.add_argument('--eval_step', type=int, default=0, help='评估模型频率，0为训练时禁用')
parser.add_argument('--num_images', type=int, default=100, help='生成图像数量')  # 减少默认值，避免生成过多
parser.add_argument('--num_step', type=int, default=1000, help='采样步数')
parser.add_argument('--pre_trained_path', default='./pth/1224_4T.pt', help='预训练模型路径')

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def set_range(x):
    return 2 * x - 1.


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x


def warmup_lr(step):
    return min(step, args.warmup) / args.warmup


def evaluate(sampler, model):
    """训练阶段评估函数（仅生成并保存图像）"""
    model.eval()
    with torch.no_grad():
        images = []
        desc = "generating evaluation images"
        # 创建保存目录
        save_dir = os.path.join(args.logdir, 'train_eval_samples')
        os.makedirs(save_dir, exist_ok=True)

        for i in trange(0, args.num_images, args.batch_size, desc=desc):
            batch_size = min(args.batch_size, args.num_images - i)
            x_T = torch.randn((batch_size, args.img_ch, args.img_size, args.img_size))
            batch_images = sampler(x_T.to(device)).cpu()
            images.append((batch_images + 1) / 2)  # 归一化到[0,1]

            # 保存当前批次的网格图
            batch_grid = make_grid(batch_images[:32], nrow=8)  # 最多显示32张
            batch_save_path = os.path.join(save_dir, f'eval_batch_{i}.png')
            save_image(batch_grid, batch_save_path)

        # 保存所有生成图像的总网格图
        all_images = torch.cat(images, dim=0)
        total_grid = make_grid(all_images[:256], nrow=16)  # 限制最大显示数量
        total_save_path = os.path.join(save_dir, 'all_eval_samples.png')
        save_image(total_grid, total_save_path)
        print(f"训练阶段评估图像已保存至: {total_save_path}")

    model.train()
    return (0.0, 0.0), 0.0, all_images  # 返回占位值，不影响训练流程


def train():
    # 数据集加载
    if args.dataset == 'cifar10':
        dataset = CIFAR10(
            root='/home/dataset/Cifar10', train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))

    elif args.dataset == 'celeba':
        SetRange = torchvision.transforms.Lambda(lambda X: 2 * X - 1.)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.CenterCrop(148),
            torchvision.transforms.Resize((64, 64)),
            torchvision.transforms.ToTensor(),
            SetRange])
        dataset = torchvision.datasets.ImageFolder(root='/home/dataset/CelebA/celeba',
                                                   transform=transform)

    elif args.dataset == 'fashion-mnist':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(set_range)
        ])
        dataset = torchvision.datasets.FashionMNIST(root='/home/dataset/FashionMnist',
                                                    train=True,
                                                    download=True,
                                                    transform=transform)

    elif args.dataset == 'mnist':
        SetRange = torchvision.transforms.Lambda(lambda X: 2 * X - 1.)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
            SetRange])
        dataset = torchvision.datasets.MNIST(root='/home/dataset/Mnist',
                                             train=True,
                                             download=True,
                                             transform=transform)

    elif args.dataset == 'lsun':
        # dataset = LSUNBed()  # 若有LSUN数据集加载逻辑可补充
        raise NotImplementedError("LSUN数据集加载逻辑未实现")
    else:
        raise NotImplementedError(f"不支持的数据集: {args.dataset}")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.num_workers, drop_last=True)

    datalooper = infiniteloop(dataloader)

    print(f'-------开始加载 {args.dataset} 数据集!-------')

    # 模型设置
    net_model = Spk_UNet(
        T=args.T, ch=args.ch, ch_mult=args.ch_mult, attn=args.attn,
        num_res_blocks=args.num_res_blocks, dropout=args.dropout, timestep=args.timestep, img_ch=args.img_ch)
    optim = torch.optim.Adam(net_model.parameters(), lr=args.lr)

    if args.resume:
        ckpt = torch.load(os.path.join(args.resume_model))
        print(f'从 {args.resume_model} 加载预训练模型')
        net_model.load_state_dict(ckpt['net_model'], strict=True)
    else:
        print('从头开始训练')

    trainer = GaussianDiffusionTrainer(
        net_model, float(args.beta_1), float(args.beta_T), args.T).to(device)

    net_sampler = GaussianDiffusionSampler(
        net_model, float(args.beta_1), float(args.beta_T), args.T, args.img_size,
        args.mean_type, args.var_type).to(device)

    if args.parallel and torch.cuda.device_count() > 1:
        trainer = torch.nn.DataParallel(trainer)
        net_sampler = torch.nn.DataParallel(net_sampler).cuda()

    # 日志设置
    sample_dir = os.path.join(args.logdir, 'sample')
    os.makedirs(sample_dir, exist_ok=True)
    x_T = torch.randn(int(args.sample_size), int(args.img_ch), int(args.img_size), int(args.img_size))
    x_T = x_T.to(device)
    # 保存真实图像作为参考
    grid = (make_grid(next(iter(dataloader))[0][:args.sample_size]) + 1) / 2
    save_image(grid, os.path.join(sample_dir, 'groundtruth.png'))

    # 显示模型大小
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print(f'模型参数: {model_size / 1024 / 1024:.2f} M')

    # 开始训练
    with trange(args.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            # 训练步骤
            optim.zero_grad()
            x_0 = next(datalooper).to(device)
            loss = trainer(x_0.float()).mean()
            loss.backward()

            if args.wandb:
                wandb.log({'training loss': loss.item()})

            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), args.grad_clip)
            optim.step()
            pbar.set_postfix(loss=f'{loss.item():.3f}')

            # 重置SNN神经元状态
            functional.reset_net(net_model)

            # 采样并保存图像
            if args.sample_step > 0 and step % args.sample_step == 0:
                net_model.eval()
                with torch.no_grad():
                    x_0 = net_sampler(x_T)
                    grid = (make_grid(x_0) + 1) / 2
                    path = os.path.join(sample_dir, f'{step}.png')
                    save_image(grid, path)
                    if args.wandb:
                        wandb.log({'sample': [wandb.Image(grid, caption=f'step {step}')]})
                net_model.train()

            # 保存模型检查点
            if args.save_step > 0 and step % args.save_step == 0 and step > 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'optim': optim.state_dict(),
                    'step': step,
                    'x_T': x_T,
                }
                save_path = os.path.join(args.logdir, f'{step}_ckpt.pt')
                torch.save(ckpt, save_path)
                print(f'模型已保存至: {save_path}')

            # 训练阶段评估（仅生成图像）
            if args.eval_step > 0 and step % args.eval_step == 0 and step > 0:
                _, _, _ = evaluate(net_sampler, net_model)
                pbar.write(f"第 {step} 步评估图像已生成并保存")


def eval():
    """单独评估函数（仅生成并保存图像）"""
    # 模型设置
    model = Spk_UNet(
        T=args.T, ch=args.ch, ch_mult=args.ch_mult, attn=args.attn,
        num_res_blocks=args.num_res_blocks, dropout=args.dropout, timestep=args.timestep, img_ch=args.img_ch)

    ckpt_path = args.pre_trained_path
    ckpt1 = torch.load(ckpt_path, map_location=torch.device('cpu'))['net_model']
    print(f'成功加载模型: {ckpt_path}')

    model.load_state_dict(ckpt1)
    functional.reset_net(model)  # 重置神经元状态
    model.eval()

    # 初始化采样器
    sampler = GaussianaussianDiffusionSampler(
        model, float(args.beta_1), float(args.beta_T), args.T, img_size=int(args.img_size),
        mean_type=args.mean_type, var_type=args.var_type, sample_type=args.sample_type, sample_steps=args.num_step).to(
        device)

    with torch.no_grad():
        images = []
        desc = "generating images"
        # 创建保存目录（按当前时间命名，避免覆盖）
        import datetime
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(args.logdir, f'eval_samples_{current_time}')
        os.makedirs(save_dir, exist_ok=True)
        print(f"生成的图像将保存至: {save_dir}")

        for i in trange(0, args.num_images, args.batch_size, desc=desc):
            batch_size = min(args.batch_size, args.num_images - i)
            x_T = torch.randn((batch_size, int(args.img_ch), int(args.img_size), int(args.img_size)))

            functional.reset_net(model)  # 每批次重置神经元状态
            batch_images = sampler(x_T.to(device)).cpu()  # 生成图像
            images.append((batch_images + 1) / 2)  # 归一化到[0,1]

            # 保存当前批次的网格图
            batch_grid = make_grid(batch_images[:32], nrow=8)  # 8列排列，最多32张
            batch_save_path = os.path.join(save_dir, f'batch_{i:04d}_{i + batch_size:04d}.png')
            save_image(batch_grid, batch_save_path)

        # 保存所有生成图像的总网格图
        all_images = torch.cat(images, dim=0)
        print(f"生成图像总数: {all_images.shape[0]}")

        # 分批次保存总网格图（避免图像过多导致内存不足）
        total_batch_size = 256  # 每批最多显示256张
        for j in range(0, len(all_images), total_batch_size):
            batch_slice = all_images[j:j + total_batch_size]
            total_grid = make_grid(batch_slice, nrow=16)  # 16列排列
            total_save_path = os.path.join(save_dir, f'all_samples_part_{j // total_batch_size}.png')
            save_image(total_grid, total_save_path)
            print(f"总网格图已保存: {total_save_path}")


def main():
    if args.wandb:
        wandb.init(project="spike_diffusion", name=f"{args.dataset}_{args.sample_type}")
        warnings.simplefilter(action='ignore', category=FutureWarning)

    seed_everything(args.seed)
    if args.train:
        train()
    if args.eval:
        eval()
    if not args.train and not args.eval:
        print('请添加 --train 或 --eval 参数以执行相应任务')


if __name__ == '__main__':
    main()
