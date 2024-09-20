import os
import time
import json
import argparse
from datetime import datetime
import torch.nn as nn
import torch.backends.cudnn as cudnn
from pathlib import Path
from functools import partial
import pandas as pd

import utils
from spiking_neuron.neuron import LIFNode
from spiking_neuron.PLIF import ParametricLIFNode
from spiking_neuron.TCLIF import TCLIFNode
from spiking_neuron.ALIF import ALIF
from load_dataset import load_dataset
from models.fc import ffMnist, fbMnist, AlexNet, ResNet
from models.CNN import SimpleCNN
from models.RNN import SimpleRNN
from utils import *


# 训练函数，用于在训练集上训练模型
def train(train_loader, model, criterion, optimizer, epoch, args):
    # 记录每个批次的时间和数据加载时间，以及损失和准确率
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()  # 设置模型为训练模式

    end = time.time()
    for i, (attributes, target) in enumerate(train_loader):
        # 记录数据加载时间
        data_time.update(time.time() - end)
        attributes = attributes.cuda(non_blocking=True)  # 将数据加载到GPU
        target = target.cuda(non_blocking=True)

        input_data = attributes.view(-1, args.time_window, 1)  # 重新调整数据形状以适应模型的输入

        reset_states(model=model)  # 重置模型的状态
        output = model(input_data)  # 前向传播
        loss = criterion(output, target)  # 计算损失

        # 计算准确率并记录损失
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), attributes.size(0))
        top1.update(acc1[0], attributes.size(0))
        top5.update(acc5[0], attributes.size(0))

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录每个批次的时间
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)  # 每隔一定次数打印训练进度

    # 记录每个epoch的训练信息
    logging.info(
        'Train Epoch: [{}/{}], lr: {:.6f}, top1: {:.4f}'.format(epoch, args.epochs, optimizer.param_groups[0]['lr'],
                                                                top1.avg))
    return top1.avg, losses.avg


# 验证函数，用于在验证集上评估模型
def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    model.eval()  # 设置模型为评估模式

    with torch.no_grad():
        end = time.time()
        for i, (attributes, target) in enumerate(val_loader):
            attributes = attributes.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            input_data = attributes.view(-1, args.time_window, 1)  # 重新调整数据形状以适应模型的输入

            reset_states(model=model)  # 重置模型状态
            output = model(input_data)  # 前向传播
            loss = criterion(output, target)  # 计算损失

            # 计算准确率并记录损失
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), attributes.size(0))
            top1.update(acc1[0], attributes.size(0))
            top5.update(acc5[0], attributes.size(0))

            # 记录每个批次的时间
            batch_time.update(time.time() - end)
            end = time.time()

    return top1.avg, top5.avg  # 返回验证集上的平均准确率


# 准确率计算函数，用于计算top-k准确率
def accuracy(output, target, topk=(1,)):
    """计算指定top-k值的准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)  # 选择top-k预测值
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # 判断预测是否正确

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))  # 计算并返回准确率
        return res


# 解析命令行参数
parser = argparse.ArgumentParser(description='Valve Data Training')
parser.add_argument('--task', default='Valve', type=str, help='Valve data training')  # 任务名称
parser.add_argument('--optim', default='adam', type=str, help='optimizer (default: adam)')  # 优化器选择
parser.add_argument('--results-dir', default='', type=str, metavar='PATH',
                    help='path to cache (default: none)')  # 结果保存路径
parser.add_argument('-p', '--print-freq', default=100, type=int, metavar='N',
                    help='print frequency (default: 10)')  # 打印频率
parser.add_argument('--seed', default=0, type=int, metavar='N', help='seed')  # 随机种子
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')  # 训练周期
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float, metavar='LR', help='initial learning rate',
                    dest='lr')  # 初始学习率
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')  # 学习率调整计划
parser.add_argument('--batch-size', default=256, type=int, metavar='N', help='mini-batch size')  # 批次大小
parser.add_argument('--wd', default=0, type=float, metavar='W', help='weight decay')  # 权重衰减
parser.add_argument("--workers", type=int, default=0)  # 数据加载的工作线程数量
parser.add_argument('--cos', action='store_true', default=False, help='use cosine lr schedule')  # 是否使用余弦学习率调度
parser.add_argument('--train-ratio', default=0.8, type=float, help='proportion of the dataset to include in the train split')  # 新增的参数

# SNN（脉冲神经网络）选项
parser.add_argument('--time-window', default=201, type=int, help='input dimension for time series data')  # 时间窗口，即输入维度
parser.add_argument('--threshold', default=1.0, type=float, help='')  # 神经元发放阈值
parser.add_argument('--detach-reset', action='store_true', default=False, help='')  # 是否使用detach重置
parser.add_argument('--hard-reset', action='store_true', default=False, help='')  # 是否使用硬重置
parser.add_argument('--decay-factor', default=1.0, type=float, help='')  # 衰减因子
parser.add_argument('--beta1', default=0., type=float, help='')  # beta1参数
parser.add_argument('--beta2', default=0., type=float, help='')  # beta2参数
parser.add_argument('--gamma', default=0.5, type=float, help='dendritic reset scaling hyper-parameter')  # 树突重置的超参数gamma
parser.add_argument('--sg', default='gau', type=str, help='sg: triangle, exp, gau, rectangle and sigmoid')  # 选择替代函数
parser.add_argument('--neuron', default='tclif', type=str, help='neuron: tclif, lif, alif and plif')  # 选择神经元模型
parser.add_argument('--network', default='resnet', type=str,
                    help='network type (options: ff, fb, alexnet, resnet, cnn, rnn)')  # 选择网络类型（前馈、反馈、AlexNet 或 ResNet）
parser.add_argument('--ind', default=1, type=int, help='input dim: 1, 4, 8')  # 输入维度
parser.add_argument('--dataset-path', default=str((Path(__file__).parent / '../../RDP/Raw_dataset').resolve()), type=str, help='Path to the dataset')
args = parser.parse_args()

# 设定保存结果的统一目录
base_results_dir = './results'
args.results_dir = os.path.join(base_results_dir, 'cs-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

Path(args.results_dir).mkdir(parents=True, exist_ok=True)  # 创建结果保存目录
logger = setup_logging(os.path.join(args.results_dir, "log-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".txt"))  # 设置日志记录

gpu = torch.device('cuda')  # 设置GPU设备
seed_everything(seed=args.seed, is_cuda=True)  # 设置随机种子

torch.backends.cudnn.benchmark = True  # 加速卷积操作

# 构建数据加载器
train_loader, test_loader, num_classes = load_dataset(dataset='ValveDataset',
                                                      batch_size=args.batch_size,
                                                      dataset_path=args.dataset_path,
                                                      is_cuda=True,
                                                      num_workers=args.workers,
                                                      train_ratio=args.train_ratio)  # 修改后的调用
args.time_window = 201  # 设置时间窗口，即输入维度
in_dim = args.ind  # 设置输入的维度

# 根据选择的替代函数来导入相应的模块
if args.sg == 'exp':
    from surrogate import SingleExponential as SG
elif args.sg == 'triangle':
    from surrogate import Triangle as SG
elif args.sg == 'rectangle':
    from surrogate import Rectangle as SG
elif args.sg == 'sigmoid':
    from surrogate import sigmoid as SG
elif args.sg == 'gau':
    from surrogate import ActFun_adp as SG
else:
    raise NotImplementedError  # 如果指定的替代函数未实现，则抛出错误

# 根据选择的神经元模型来初始化相应的神经元类
node = None
if args.neuron == 'lif':
    node = LIFNode
elif args.neuron == 'tclif':
    node = TCLIFNode
elif args.neuron == 'plif':
    node = ParametricLIFNode
elif args.neuron == 'alif':
    node = ALIF

# 初始化可学习的beta参数
beta = torch.full([1, 2], 0., dtype=torch.float)  # 创建一个形状为[1, 2]的张量，初始值为0
beta[0][0] = args.beta1  # 设置beta1
beta[0][1] = args.beta2  # 设置beta2
init1 = torch.sigmoid(beta[0][0]).cpu().item()  # 初始化beta1
init2 = torch.sigmoid(beta[0][1]).cpu().item()  # 初始化beta2
print("beta init from {:.2f} and {:.2f}".format(-init1, init2))  # 打印初始化的beta值

# 构建脉冲神经元的参数字典
spk_params = {"time_window": args.time_window,
              'v_threshold': args.threshold,
              'surrogate_function': SG.apply,
              'hard_reset': False,
              'detach_reset': False,
              'decay_factor': beta,
              'gamma': args.gamma}

# 使用partial来生成特定的神经元实例
spiking_neuron = partial(node,
                         v_threshold=spk_params['v_threshold'],
                         surrogate_function=spk_params['surrogate_function'],
                         hard_reset=spk_params['hard_reset'],
                         detach_reset=spk_params['detach_reset'],
                         decay_factor=spk_params['decay_factor'],
                         gamma=spk_params['gamma'])

# 根据指定的网络类型来初始化模型
if args.task == 'Valve':
    if args.network == 'ff':
        model = ffMnist(in_dim=in_dim, spiking_neuron=spiking_neuron).to(gpu)  # 前馈神经网络
    elif args.network == 'fb':
        model = fbMnist(in_dim=in_dim, spiking_neuron=spiking_neuron).to(gpu)  # 反馈神经网络
    elif args.network == 'alexnet':
        model = AlexNet(in_dim=in_dim, spiking_neuron=spiking_neuron).to(gpu)  # AlexNet
    elif args.network == 'resnet':
        model = ResNet(in_dim=in_dim, spiking_neuron=spiking_neuron).to(gpu)  # ResNet
    elif args.network == 'cnn':
        model = SimpleCNN(in_dim=args.time_window).to(gpu)
    elif args.network == 'rnn':
        model = SimpleRNN(in_dim=args.time_window).to(gpu)
else:
    raise NotImplementedError  # 如果任务未实现，则抛出错误

logging.info(str(model))  # 记录模型结构
para = utils.count_parameters(model)  # 计算并记录模型参数数量

criterion = nn.CrossEntropyLoss().cuda(gpu)  # 使用交叉熵作为损失函数

# 选择优化器
if args.optim == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
elif args.optim == 'adam':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
else:
    raise NotImplementedError  # 如果指定的优化器未实现，则抛出错误

# 保存命令行参数配置
with open(os.path.join(args.results_dir, 'args.json'), 'w') as fid:
    json.dump(args.__dict__, fid, indent=2)
logging.info(str(args))  # 记录参数配置

start_epoch = 0  # 初始化起始epoch

# 调整打印频率
if args.print_freq > len(train_loader):
    args.print_freq = math.ceil(len(train_loader) // 2)

best_acc = argparse.Namespace(top1=0, top5=0)  # 初始化最佳准确率
# 初始化用于存储训练和测试结果的DataFrame
train_res = pd.DataFrame()
test_res = pd.DataFrame()
best = 0  # 初始化最佳准确率

# 开始训练和验证循环
for epoch in range(start_epoch, args.epochs):
    flag = False
    adjust_learning_rate(optimizer, epoch, args)  # 调整学习率

    train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch, args)  # 训练模型

    acc1, acc5 = validate(test_loader, model, criterion, args)  # 验证模型
    best_acc.top1 = max(best_acc.top1, acc1)  # 更新最佳Top-1准确率
    best_acc.top5 = max(best_acc.top5, acc5)  # 更新最佳Top-5准确率
    train_res[str(epoch)] = [train_acc.cpu().item(), train_loss]  # 记录训练结果
    test_res[str(epoch)] = [acc1.cpu().item()]  # 记录验证结果

    if acc1.cpu().item() >= best:
        flag = True
        best = acc1.cpu().item()  # 更新最佳准确率
    print('Test Epoch: [{}/{}], lr: {:.6f}, acc: {:.4f}, best: {:.4f}'.format(epoch, args.epochs,
                                                                                     optimizer.param_groups[0]['lr'],
                                                                                     acc1, best))

    train_res.to_csv(os.path.join(args.results_dir, 'train_res.csv'), index=True)  # 保存训练结果到CSV
    test_res.to_csv(os.path.join(args.results_dir, 'test_res.csv'), index=True)  # 保存测试结果到CSV

    # 保存检查点
    save_checkpoint({
        'epoch': epoch + 1,
        'best_acc': best_acc,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, is_best=flag, dirname=args.results_dir, filename='checkpoint.pth.tar')
