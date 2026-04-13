import os
import sys
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import transforms

from spikingjelly.datasets.n_mnist import NMNIST
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.n_caltech101 import NCaltech101
from spikingjelly.datasets import split_to_train_test_set
from spikingjelly.activation_based import functional, encoding, neuron, surrogate, layer

from model import VGG, Fusion
from utils import *

import torch.serialization

torch.serialization.add_safe_globals([torch.utils.data.dataset.TensorDataset])

import glob

def _load_freq_weights_from_pt(path: str):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(obj, dict):
        if 'final_weights' in obj:
            w = obj['final_weights']
        elif 'weights' in obj:
            w = obj['weights']
        else:
            raise ValueError(f'未在{path}中找到final_weights或weights')
    else:
        raise ValueError(f'不支持的pt格式: {type(obj)}')
    import numpy as np
    return np.array(w, dtype=float)

def _auto_find_latest_complete_history(args):
    # --- 优化建议：修正自动搜索路径以匹配新的 "runs/full" 结构 ---
    # 它现在会根据核心参数在 "runs/full" 目录下搜索所有匹配的训练历史，并返回最新的一个。
    search_pattern = os.path.join(args.run_dir, 'full', f'{args.dataset}_{args.model}_{args.act}_T{args.T}_*', 'freq_weights', 'complete_freq_weight_history_*.pt')
    cands = sorted(glob.glob(search_pattern), key=os.path.getmtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f'自动搜索权重历史失败，未在以下模式中找到文件: {search_pattern}')
    return cands[0]

def _gen_keep_indices_by_ratio(weights, keep_ratio=None, drop_ratio=None):
    import numpy as np
    n = len(weights)
    keep_n = max(1, min(n, int(round(n * (keep_ratio if keep_ratio is not None else (1 - (drop_ratio or 0.0)))))))
    order_desc = np.argsort(-weights)
    keep_idx = np.sort(order_desc[:keep_n])
    kth_val = weights[order_desc[keep_n - 1]]
    return keep_idx, keep_n, kth_val

def _save_keep_indices(args, keep_idx, keep_n, weights, src_file, out_path=None):
    import numpy as np
    meta = {
        'dataset': args.dataset, 'arch': args.model, 'act': args.act, 'T': args.T,
        'total_bins': int(len(weights)), 'keep_count': int(keep_n), 'keep_ratio': float(keep_n / len(weights)),
        'weights_mean': float(np.mean(weights)), 'weights_std': float(np.std(weights)),
        'weights_min': float(np.min(weights)), 'weights_max': float(np.max(weights)),
        'src_weight_file': src_file, 'indices_dtype': 'int64'
    }
    # --- 优化建议：直接使用args.out_dir，不再硬编码路径 ---
    save_dir = args.out_dir # 使用在主逻辑中定义的目录
    os.makedirs(save_dir, exist_ok=True)
    if out_path is None:
        out_path = os.path.join(save_dir, f'{args.dataset}_{args.model}_{args.act}_T{args.T}_freq_keep_idx_{keep_n}of{len(weights)}.pt')
    torch.save({'keep_indices': torch.tensor(keep_idx, dtype=torch.long), 'meta': meta}, out_path)
    return out_path


############## Reproducibility ##############
############## Reproducibility ##############
seed = int(os.environ.get('SEED', 2023))
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#############################################
#############################################

parser = argparse.ArgumentParser(description='EC-SNN interface for all experiments')

parser.add_argument('-num_cls', default=10, type=int, help='number of class for classification')
# --- 优化建议：移除旧的、令人困惑的目录参数，并添加一个统一的根目录参数 ---
parser.add_argument('-run_dir', type=str, default='./runs', help='Root directory for all outputs (logs, weights, etc.)')
parser.add_argument('-data_dir', type=str, default='.', help='root dir of dataset')
parser.add_argument('-device', default='cpu', help='device')  # cuda:0
parser.add_argument('-j', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-dataset', default='cifar10', help='dataset name', choices=['mnist', 'cifar10', 'caltech', 'cifar10_dvs', 'nmnist', 'ncaltech', 'gtzan', 'urbansound', 'esc50'])
# Use a single, clear argument for the model architecture
parser.add_argument('-model', default='cifarnet_snn', help='model architecture, e.g., vgg9_snn, cifarnet_snn')
parser.add_argument('-act', default='snn', help='ANN or SNN, default is snn, determine relu or lif for spikes')

parser.add_argument('-T', default=5, type=int, help='simulating time-steps')
parser.add_argument('-b', default=128, type=int, help='batch size')
parser.add_argument('-epochs', default=70, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-opt', default='adam', type=str, help='use which optimizer. SDG or Adam')
parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
parser.add_argument('-lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('-tau', default=4./3, type=float, help='parameter tau of LIF neuron')

# --- 优化建议：添加一个命令行开关来控制是否使用频率权重 ---
parser.add_argument('-use_freq_weights', action='store_true', help='Enable learnable frequency weights for pruning.')

parser.add_argument('-encode', default='r', type=str, help='spike encoding methode. (p)Poisson, (l)Latency, or (r)Raw')
parser.add_argument('-train', action='store_true', help='generate origin trained model')
parser.add_argument('-prune', action='store_true', help='generate pruned trained model')
parser.add_argument('-fusion', action='store_true', help='fusion the final results with all networks in split_dir')
parser.add_argument('-infer', action='store_true', help='get infer time result')
parser.add_argument('-energy', action='store_true', help='get energy consumption result')
parser.add_argument('-split', action='store_true', help='split mode or single mode')


parser.add_argument('-apoz', '--apoz_threshold', type=float, default=93., help="APOZ threshold for filter's activation map")
parser.add_argument('-c', '--split_class', type=int, nargs='+', help='class code for splitting')
parser.add_argument('-min_f', type=int, default=16, help="minimym number of filters for specific conv-layer after split")

# 修正参数定义，统一使用单横线格式
parser.add_argument('-fw_lr_mul', type=float, default=10.0, help='freq-weights 学习率放大倍数')
parser.add_argument('-fw_grad_mul', type=float, default=1.0, help='freq-weights 反向梯度放大倍数')
parser.add_argument('-fw_init_std', type=float, default=0.2, help='freq-weights 初始化扰动标准差')

parser.add_argument('-freq_prune', action='store_true', help='按频率权重分位生成保留索引并退出')
parser.add_argument('-freq_weights_file', type=str, default=None, help='频率权重pt文件路径')
parser.add_argument('-keep_ratio', type=float, default=None, help='保留比例，如0.75；优先于drop_ratio')
parser.add_argument('-drop_ratio', type=float, default=0.25, help='裁剪比例，如0.25代表裁掉25%')
parser.add_argument('-freq_index_out', type=str, default=None, help='保留索引保存路径')
parser.add_argument('-freq_keep_idx', type=str, default=None, help='已生成的频率保留索引pt文件路径（用于训练/推理阶段裁剪输入）')

parser.add_argument('-random_gen', action='store_true', help='生成随机剪枝索引（用于Baseline对比）')
parser.add_argument('-variance_gen', action='store_true', help='按频率维度方差排序生成保留索引（Variance Baseline）')
parser.add_argument('-lowpass_gen', action='store_true', help='保留前K个低频bin生成保留索引（Fixed Low-pass Baseline）')

# --- 为Mixup和学习率调度器新增参数 ---
parser.add_argument('-mixup', action='store_true', help='use mixup data augmentation')
parser.add_argument('-mixup_alpha', default=0.2, type=float, help='mixup alpha value')
parser.add_argument('-scheduler', default='cosine', type=str, help='learning rate scheduler (cosine or step)')


args = parser.parse_args()
# --- 优化建议：在这里为args动态添加路径属性，避免混淆 ---
args.out_dir = None      # 将用于存放日志和当前运行的所有输出
args.model_dir = None    # 将用于存放模型权重
args.split_dir = None    # 仅为旧的APOZ/Fusion功能保留

desc = ''
if args.train:
    desc += 'train'
if args.prune:
    desc += 'prune'
if args.infer:
    desc += 'infer'
if args.fusion:
    desc += 'fusion'
if args.energy:
    desc += 'energy'
if args.random_gen:
    desc = 'random_gen'
if args.variance_gen:
    desc = 'variance_gen'
if args.lowpass_gen:
    desc = 'lowpass_gen'
if args.split:
    assert (args.split and args.energy) or (args.split and args.infer), 'arg split must be paired with arg energy or infer'
    desc = 'split_' + desc

# --- 优化建议：重构整个目录设置逻辑 ---
import datetime as _dt
_ts = _dt.datetime.now().strftime('%Y%m%d-%H%M%S')

# 1. 推断当前运行步骤
_step = 'misc' # Default step
if args.freq_prune:
    _step = 'prune'
elif args.train and args.freq_keep_idx is None:
    _step = 'full'
elif args.train and args.freq_keep_idx is not None:
    _step = 'pruned'

# 2. 获取剪枝标签 (e.g., 'keep30of40')
_keep_tag = 'full'
if args.freq_keep_idx is not None:
    try:
        _obj_ki = torch.load(args.freq_keep_idx, map_location='cpu', weights_only=False)
        _keep_n = int(_obj_ki['keep_indices'].numel())
        _total_bins = img_size_ref[args.dataset][0]
        _keep_tag = f'keep{_keep_n}of{_total_bins}'
    except Exception:
        _keep_tag = 'kept'  # Fallback

# 3. 根据步骤设置清晰的输出路径
if _step == 'full':
    run_name = f'{args.dataset}_{args.model}_{args.act}_T{args.T}_b{args.b}_lr{args.lr}_{_ts}'
    args.out_dir = os.path.join(args.run_dir, 'full', run_name)
    args.model_dir = os.path.join(args.out_dir, 'weights')
elif _step == 'pruned':
    # --- 优化建议：同样从输入路径中提取源信息，保持命名一致性 ---
    source_run_name = "unknown_source"
    if args.freq_keep_idx:
        try:
            # 路径格式: .../prune/SOURCE_RUN_NAME__0.25drop/FILE.pt
            # 我们旨在提取 SOURCE_RUN_NAME
            full_source_name = os.path.basename(os.path.dirname(args.freq_keep_idx))
            source_run_name = full_source_name.split('__')[0]
        except Exception:
            source_run_name = f'{args.dataset}_{args.model}_{_ts}'
    
    run_name = f'{source_run_name}__{_keep_tag}_b{args.b}_lr{args.lr}_{_ts}'
    args.out_dir = os.path.join(args.run_dir, 'pruned', run_name)
    args.model_dir = os.path.join(args.out_dir, 'weights')
elif _step == 'prune':
    # --- 优化建议：从权重文件路径中提取源训练名称，使命名更具追溯性 ---
    source_run_name = "unknown_source"
    # 我们优先从用户提供的`-freq_weights_file`参数中提取信息
    if args.freq_weights_file:
        try:
            # 路径格式为: .../runs/full/RUN_NAME/freq_weights/FILE.pt
            # 我们旨在提取 RUN_NAME
            source_run_name = os.path.basename(os.path.dirname(os.path.dirname(args.freq_weights_file)))
        except Exception:
            # 如果路径解析失败，则使用备用命名方案
            source_run_name = f'{args.dataset}_{args.model}_{_ts}'
    
    run_name = f'{source_run_name}__{args.drop_ratio}drop'
    args.out_dir = os.path.join(args.run_dir, 'prune', run_name)
    args.model_dir = os.path.join(args.out_dir, 'weights')
else:
    # 为其他模式（如仅推理、能量分析）提供一个默认输出目录
    args.out_dir = os.path.join(args.run_dir, 'misc', desc, _ts)

# 为旧的APOZ/Fusion功能设置一个隔离的目录，使其不再与新逻辑混淆
args.split_dir = os.path.join(args.run_dir, 'legacy_apoz_fusion')

# 4. 创建所需目录
# 确保所有可能用到的目录都被创建
os.makedirs(args.out_dir, exist_ok=True)
if args.model_dir:
    os.makedirs(args.model_dir, exist_ok=True)
# --- 目录重构结束 ---

# 新增：启用cuDNN benchmark优化小特征图卷积内核选择
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.deterministic = False
# 允许TF32以提升矩阵与卷积吞吐（对数值影响极小，Ada/4090友好）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- 优化建议：现在路径已经固定，在这里创建唯一的Logger实例 ---
logger = Logger(args, desc=desc)

logger.info(args)

if args.freq_prune:
    logger.info('开始按分位生成频率保留索引')
    src_file = args.freq_weights_file or _auto_find_latest_complete_history(args)
    logger.info(f'使用权重文件: {src_file}')
    weights = _load_freq_weights_from_pt(src_file)
    import numpy as np
    logger.info(f'权重统计: mean={np.mean(weights):.4f}, std={np.std(weights):.4f}, range=[{np.min(weights):.4f}, {np.max(weights):.4f}], len={len(weights)}')
    keep_idx, keep_n, kth = _gen_keep_indices_by_ratio(weights, keep_ratio=args.keep_ratio, drop_ratio=args.drop_ratio)
    logger.info(f'保留 {keep_n}/{len(weights)} ({keep_n/len(weights)*100:.2f}%), Top-K 门限≈{kth:.4f}')
    out_path = _save_keep_indices(args, keep_idx, keep_n, weights, src_file, args.freq_index_out)
    logger.info(f'频率保留索引已保存: {out_path}')
    sys.exit(0)

if args.random_gen:
    logger.info('----------------------------------------------------------------')
    logger.info('开始生成随机频率保留索引 (Random Pruning Baseline)')
    logger.info('注意：此模式用于生成对比实验的基准，完全忽略频率权重，随机选择保留的频率。')
    
    import numpy as np
    
    # 1. 确定总频率数 (Total Bins)
    if args.dataset not in img_size_ref:
         raise ValueError(f"未知数据集 {args.dataset}，无法确定频率维度")
    total_bins = img_size_ref[args.dataset][0]
    
    # 2. 计算需要保留的数量 (Keep N)
    keep_ratio = args.keep_ratio
    drop_ratio = args.drop_ratio
    # 逻辑与 _gen_keep_indices_by_ratio 保持一致
    keep_n = max(1, min(total_bins, int(round(total_bins * (keep_ratio if keep_ratio is not None else (1 - (drop_ratio or 0.0)))))))
    
    # 3. 生成随机索引
    # 使用 Numpy 的随机抽样，replace=False 保证不重复
    rng = np.random.default_rng() # 使用新的随机生成器
    random_indices = rng.choice(total_bins, keep_n, replace=False)
    
    # 4. 排序索引
    # 虽然是随机选择，但为了保持频谱的空间结构（低频到高频的顺序），通常需要对索引进行排序
    keep_idx = np.sort(random_indices)
    
    # 5. 构造保存路径
    save_dir = args.out_dir
    os.makedirs(save_dir, exist_ok=True)
    # 文件名包含 random 标识
    out_path = os.path.join(save_dir, f'{args.dataset}_RANDOM_keep_{keep_n}of{total_bins}.pt')
    
    # 6. 保存文件
    # 构造与 FW-SNN 兼容的字典格式
    meta = {
        'dataset': args.dataset, 
        'method': 'random_baseline',
        'total_bins': total_bins, 
        'keep_count': keep_n,
        'keep_ratio': keep_n / total_bins,
        'indices_dtype': 'int64',
        'timestamp': _ts
    }
    
    torch.save({'keep_indices': torch.tensor(keep_idx, dtype=torch.long), 'meta': meta}, out_path)
    
    logger.info(f'随机保留数量: {keep_n}/{total_bins} ({(keep_n/total_bins)*100:.2f}%)')
    logger.info(f'随机索引已保存至: {out_path}')
    logger.info(f'保留的索引内容: {keep_idx}')
    logger.info('生成完毕，程序退出。请使用生成的 .pt 文件路径配合 -freq_keep_idx 参数进行训练。')
    logger.info('----------------------------------------------------------------')
    sys.exit(0)

if args.variance_gen:
    logger.info('----------------------------------------------------------------')
    logger.info('开始生成方差频率保留索引 (Variance-based Pruning Baseline)')
    logger.info('策略：计算训练集中每个频率bin的方差，保留方差最大（信息量最丰富）的top-K个bin。')

    import numpy as np

    if args.dataset not in img_size_ref:
        raise ValueError(f"未知数据集 {args.dataset}，无法确定频率维度")
    total_bins = img_size_ref[args.dataset][0]

    keep_ratio = args.keep_ratio
    drop_ratio = args.drop_ratio
    keep_n = max(1, min(total_bins, int(round(total_bins * (keep_ratio if keep_ratio is not None else (1 - (drop_ratio or 0.0)))))))

    # 加载缓存的数据集以计算方差
    _ds_cache = {
        'urbansound': './urbansound_dataset.pt',
        'gtzan': './gtzan_dataset.pt',
        'esc50': './esc50.pt',
    }
    _cache_path = _ds_cache.get(args.dataset)
    if _cache_path is None or not os.path.exists(_cache_path):
        raise FileNotFoundError(f'请先运行一次 -train 以生成数据集缓存文件: {_cache_path}')

    logger.info(f'加载数据集缓存: {_cache_path}')
    _train_ds, _ = torch.load(_cache_path, map_location='cpu', weights_only=False)

    # TensorDataset: .tensors[0] 即 X, shape (N, 1, F, T)
    if hasattr(_train_ds, 'tensors'):
        _all_X = _train_ds.tensors[0]  # (N, C, F, T)
    elif hasattr(_train_ds, 'dataset') and hasattr(_train_ds.dataset, 'transform'):
        # Subset 类型（如 ESC-50），逐样本采集
        _samples = [_train_ds[i][0] for i in range(len(_train_ds))]
        _all_X = torch.stack(_samples)
    else:
        raise TypeError(f'无法从 {type(_train_ds)} 中提取频率数据')

    # 沿 H（频率）维度计算方差：先把 (N,C,F,T) 展平为 (F, -1)，再算每个 bin 的方差
    _X_np = _all_X.numpy()
    _freq_var = np.var(_X_np, axis=(0, 1, 3))  # shape: (F,)
    logger.info(f'各频率bin方差统计: mean={np.mean(_freq_var):.6f}, std={np.std(_freq_var):.6f}, range=[{np.min(_freq_var):.6f}, {np.max(_freq_var):.6f}]')

    # 按方差从大到小排序，保留 top-K
    _order_desc = np.argsort(-_freq_var)
    keep_idx = np.sort(_order_desc[:keep_n])

    save_dir = args.out_dir
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f'{args.dataset}_VARIANCE_keep_{keep_n}of{total_bins}.pt')

    meta = {
        'dataset': args.dataset,
        'method': 'variance_baseline',
        'total_bins': total_bins,
        'keep_count': keep_n,
        'keep_ratio': keep_n / total_bins,
        'freq_variances': _freq_var.tolist(),
        'indices_dtype': 'int64',
        'timestamp': _ts
    }
    torch.save({'keep_indices': torch.tensor(keep_idx, dtype=torch.long), 'meta': meta}, out_path)

    logger.info(f'方差保留数量: {keep_n}/{total_bins} ({(keep_n/total_bins)*100:.2f}%)')
    logger.info(f'方差索引已保存至: {out_path}')
    logger.info(f'保留的索引内容: {keep_idx}')
    logger.info(f'被剪掉的索引: {np.sort(_order_desc[keep_n:])}')
    logger.info('生成完毕，程序退出。请使用生成的 .pt 文件路径配合 -freq_keep_idx 参数进行训练。')
    logger.info('----------------------------------------------------------------')
    sys.exit(0)

if args.lowpass_gen:
    logger.info('----------------------------------------------------------------')
    logger.info('开始生成低通滤波保留索引 (Fixed Low-pass Pruning Baseline)')
    logger.info('策略：固定保留前K个低频bin（index 0 ~ K-1），丢弃高频部分。')

    import numpy as np

    if args.dataset not in img_size_ref:
        raise ValueError(f"未知数据集 {args.dataset}，无法确定频率维度")
    total_bins = img_size_ref[args.dataset][0]

    keep_ratio = args.keep_ratio
    drop_ratio = args.drop_ratio
    keep_n = max(1, min(total_bins, int(round(total_bins * (keep_ratio if keep_ratio is not None else (1 - (drop_ratio or 0.0)))))))

    keep_idx = np.arange(keep_n, dtype=np.int64)

    save_dir = args.out_dir
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f'{args.dataset}_LOWPASS_keep_{keep_n}of{total_bins}.pt')

    meta = {
        'dataset': args.dataset,
        'method': 'lowpass_baseline',
        'total_bins': total_bins,
        'keep_count': keep_n,
        'keep_ratio': keep_n / total_bins,
        'indices_dtype': 'int64',
        'timestamp': _ts
    }
    torch.save({'keep_indices': torch.tensor(keep_idx, dtype=torch.long), 'meta': meta}, out_path)

    logger.info(f'低通保留数量: {keep_n}/{total_bins} ({(keep_n/total_bins)*100:.2f}%)')
    logger.info(f'低通索引已保存至: {out_path}')
    logger.info(f'保留的索引内容（前{keep_n}个低频bin）: {keep_idx}')
    logger.info('生成完毕，程序退出。请使用生成的 .pt 文件路径配合 -freq_keep_idx 参数进行训练。')
    logger.info('----------------------------------------------------------------')
    sys.exit(0)

if not args.infer:
    logger.info('Load data')
    if args.dataset == 'mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = torchvision.datasets.MNIST(
                root=args.data_dir,
                train=True,
                transform=transform_train,
                download=True)
        test_dataset = torchvision.datasets.MNIST(
                root=args.data_dir,
                train=False,
                transform=transform_test,
                download=True)

    elif args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(), 
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir,
            train=True,
            transform=transform_train,
            download=True)
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir,
            train=False,
            transform=transform_test,
            download=True)  
    elif args.dataset == 'caltech':
        if not os.path.exists('./caltech_dataset.pt'):
            # batch=16
            transform_all = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3,1,1) if x.shape[0] == 1 else x),
                transforms.Normalize(mean = [0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])

            dataset = torchvision.datasets.Caltech101(
                root=args.data_dir,
                transform=transform_all,
                download=True)
            dataset = CaltechTop10(dataset)
            train_dataset, test_dataset = split_to_train_test_set(0.8, dataset, args.num_cls)
            torch.save([train_dataset, test_dataset], './caltech_dataset.pt')
        else:
            print('Files already processed in data_dir')
            train_dataset, test_dataset = torch.load('./caltech_dataset.pt')

    elif args.dataset == 'cifar10_dvs':
        if not os.path.exists('./cifar10_dvs_dataset.pt'):
            dataset = CIFAR10DVS(
                root=args.data_dir, 
                data_type='frame', 
                frames_number=args.T, 
                split_by='number')
            train_dataset, test_dataset = split_to_train_test_set(0.8, dataset, args.num_cls)
            torch.save([train_dataset, test_dataset], './cifar10_dvs_dataset.pt')
        else:
            print('Files already processed in data_dir')
            train_dataset, test_dataset = torch.load('./cifar10_dvs_dataset.pt')

    elif args.dataset == 'nmnist':
        train_dataset = NMNIST(
            root=args.data_dir,
            train=True,
            data_type='frame',
            frames_number=args.T,
            split_by='number')
        test_dataset = NMNIST(
            root=args.data_dir,
            train=False,
            data_type='frame',
            frames_number=args.T,
            split_by='number')

    elif args.dataset == 'ncaltech':
        if not os.path.exists('./ncaltech_dataset.pt'):
            transform_all = transforms.Compose([
                transforms.Lambda(lambda x: torch.tensor(x)),
                transforms.Resize((224, 224), antialias=True),
            ])

            dataset = NCaltech101(
                root=args.data_dir, 
                data_type='frame',
                frames_number=args.T,
                transform=transform_all,
                split_by='number')
            
            dataset = CaltechTop10(dataset)
            
            train_dataset, test_dataset = split_to_train_test_set(0.8, dataset, args.num_cls)
            torch.save([train_dataset, test_dataset], './ncaltech_dataset.pt')
        else:
            print('Files already processed in data_dir')
            train_dataset, test_dataset = torch.load('./ncaltech_dataset.pt')

    elif args.dataset == 'gtzan':
        if not os.path.exists('./gtzan_dataset.pt'):
            train_dataset, test_dataset = get_gtzan_dataset() 
            torch.save([train_dataset, test_dataset], './gtzan_dataset.pt')
        else:
            print('Files already processed in data_dir')
            # --- 修复代码：添加 weights_only=False 参数 ---
            try:
                train_dataset, test_dataset = torch.load('./gtzan_dataset.pt', weights_only=False)
            except Exception as e:
                logger.info(f'使用 weights_only=False 加载数据集失败: {e}')
                # 如果仍然失败，尝试重新生成数据集
                logger.info('重新生成数据集...')
                train_dataset, test_dataset = get_gtzan_dataset() 
                torch.save([train_dataset, test_dataset], './gtzan_dataset.pt')
            # --- 修复代码结束 ---

    elif args.dataset == 'urbansound':
        if not os.path.exists('./urbansound_dataset.pt'):
            train_dataset, test_dataset = get_urbansound_dataset() 
            torch.save([train_dataset, test_dataset], './urbansound_dataset.pt')
        else:
            print('Files already processed in data_dir')
            # --- 修复代码：添加 weights_only=False 参数 ---
            try:
                train_dataset, test_dataset = torch.load('./urbansound_dataset.pt', weights_only=False)
            except Exception as e:
                logger.info(f'使用 weights_only=False 加载数据集失败: {e}')
                # 如果仍然失败，尝试重新生成数据集
                logger.info('重新生成数据集...')
                train_dataset, test_dataset = get_urbansound_dataset() 
                torch.save([train_dataset, test_dataset], './urbansound_dataset.pt')
            # --- 修复代码结束 ---

    elif args.dataset == 'esc50':
        args.num_cls = 50
        # 恢复与其他数据集一致的.pt缓存加载机制
        if not os.path.exists('./esc50.pt'):
            print('Loading and processing ESC-50...')
            # 修正：不再传递错误的 data_dir，让函数使用自己的默认路径
            train_dataset, test_dataset = get_esc50_optimized_dataset()
            torch.save([train_dataset, test_dataset], './esc50.pt')
        else:
            print('Loading from cached esc50.pt...')
            train_dataset, test_dataset = torch.load('./esc50.pt')

    else:
        raise NotImplementedError(f'Invalid dataset name: {args.dataset}...')

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.b,
        shuffle=True,
        drop_last=True, 
        num_workers=args.j, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2)

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.b,
        shuffle=False,
        drop_last=False, 
        num_workers=args.j, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2)

    logger.info(f'[{args.dataset}] train samples: {len(train_dataset)}, test samples: {len(test_dataset)}')

else:
    pass

# 加载频率保留索引（若提供）
keep_indices = None
if args.freq_keep_idx is not None:
    if not os.path.exists(args.freq_keep_idx):
        raise FileNotFoundError(f'freq_keep_idx 文件不存在: {args.freq_keep_idx}')
    obj = torch.load(args.freq_keep_idx, map_location='cpu', weights_only=False)
    keep_indices = obj['keep_indices'].long()
    logger.info(f'加载频率保留索引: 保留 {keep_indices.numel()} / {img_size_ref[args.dataset][0]}')

# 预先将索引移动到目标设备，避免训练循环中重复 .to()
keep_indices_device = keep_indices.to(args.device) if keep_indices is not None else None


# MODEL
# Always use args.model to determine the architecture
arch_name = args.model.split('_')[0]

if arch_name.startswith('vgg') or arch_name == 'cifarnet':
    # --- 优化建议：根据是否剪枝，动态调整传递给模型的图像尺寸 ---
    original_h, original_w = img_size_ref[args.dataset]
    # 如果存在保留索引，则使用索引数量作为新的高度，否则使用原始高度
    model_h = keep_indices.numel() if keep_indices is not None else original_h
    
    net = VGG(
        arch=arch_name, # Extracts 'vgg9' from 'vgg9_snn'
        num_cls=args.num_cls, 
        img_size=(model_h, original_w), # 传递正确的、可能已被修改的尺寸
        input_dim=input_dim_ref[args.dataset],
        act=args.act,
        tau=args.tau,
        # --- 优化建议：将命令行参数传递给模型 ---
        use_freq_weights=args.use_freq_weights
    )
else:
    # Keep cifarnet logic if needed, or raise error
    raise NotImplementedError(f"Model architecture '{arch_name}' not implemented.")

logger.info(net)


def slice_freq(x):
    # x: (N,C,H,W) 或 (N,T,C,H,W)
    if keep_indices is None:
        return x
    if x.dim() == 4:
        # 非神经数据 (N,C,H,W) 沿 H 维裁剪；裁剪后强制 contiguous
        return x.index_select(2, keep_indices_device).contiguous()
    elif x.dim() == 5:
        # 神经数据 (N,T,C,H,W) 沿 H 维裁剪（在 transpose 之前切）；裁剪后强制 contiguous
        return x.index_select(3, keep_indices_device).contiguous()
    return x

logger.info('Network Architecture Details:')
logger.info('\n' + str(net))
logger.info('Arguments Settings:')
logger.info(str(args))
logger.info('Running Command:')
logger.info(' '.join(sys.argv))

encoder = None
if args.encode == 'p':
    encoder = encoding.PoissonEncoder()
elif args.encode == 'l':
    encoder = encoding.LatencyEncoder()
elif args.encode == 'r':
    pass
else:
    raise NotImplementedError(f'invalid encoding method: {args.encode}')

if args.train:
    net.to(args.device)

    # --- 新增：仅在ANN通路启用AMP缩放器 ---
    scaler = torch.cuda.amp.GradScaler(enabled=(args.act == 'ann'))

    # --- 新增代码：频率权重记录相关变量 ---
    freq_weight_history = []  # 记录每个epoch的频率权重变化
    freq_weight_save_dir = os.path.join(args.out_dir, 'freq_weights')
    ensure_dir(freq_weight_save_dir)
    # --- 新增代码结束 ---

    optimizer = None
    fw_params = []
    base_params = []

    if hasattr(net, 'freq_weights') and net.use_freq_weights:
        fw_params = [net.freq_weights]

    for n, p in net.named_parameters():
        if fw_params and p is net.freq_weights:
            continue
        base_params.append(p)

    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(
            [{'params': base_params, 'lr': args.lr, 'momentum': args.momentum, 'weight_decay': 5e-4},
             {'params': fw_params,  'lr': args.lr * args.fw_lr_mul, 'momentum': args.momentum, 'weight_decay': 0.0}]
            if fw_params else
            [{'params': base_params, 'lr': args.lr, 'momentum': args.momentum, 'weight_decay': 5e-4}]
        )
    elif args.opt == 'adam':
        # 统一所有数据集的训练策略，移除对esc50的特殊处理
        weight_decay = 0.0
        optimizer = torch.optim.Adam(
            [{'params': base_params, 'lr': args.lr, 'weight_decay': weight_decay},
             {'params': fw_params,  'lr': args.lr * args.fw_lr_mul, 'weight_decay': 0.0}]
            if fw_params else
            [{'params': base_params, 'lr': args.lr, 'weight_decay': weight_decay}]
        )
    else:
        raise NotImplementedError(f'invalid optimizer: {args.opt}')

    # --- 新增学习率调度器 ---
    if args.scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    else: # 默认为阶梯式衰减
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    max_test_acc = -1
    train_time_record = []
    
    # --- 新增代码：添加进度条和性能监控 ---
    from tqdm import tqdm
    logger.info(f'开始训练，使用设备: {args.device}')
    if args.device == 'cpu':
        logger.info('警告：在CPU上训练SNN模型可能非常缓慢，建议使用GPU')
    # --- 新增代码结束 ---
    
    for epoch in range(args.epochs):
        logger.info(f'start epoch {epoch + 1}')
        net.train()

        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        
        # --- 新增代码：添加进度条 ---
        pbar = tqdm(train_data_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch_idx, (img, label) in enumerate(pbar):
            optimizer.zero_grad()
            # 非阻塞 H2D 搬运（配合 DataLoader pin_memory=True）
            img = img.to(args.device, non_blocking=True)
            label = label.to(args.device, non_blocking=True)

            out_fr = 0.
            loss = 0.

            # 使用AMP自动混合精度（仅ANN启用）
            with torch.cuda.amp.autocast(enabled=(args.act == 'ann')):
                if args.dataset in ['cifar10_dvs', 'nmnist', 'ncaltech']:
                    # 先裁剪，再转置到 (T,N,C,H,W)
                    img = slice_freq(img)          # (N,T,C,H,W) -> 裁H
                    img = img.transpose(0, 1)
                    for t in range(args.T):
                        output = net(img[t])
                        out_fr += output
                        loss += F.cross_entropy(output, label)
                else:
                    # 非神经数据：编码前先裁剪
                    img = slice_freq(img)          # (N,C,H,W) -> 裁H
                    for t in range(args.T):
                        encoded_img = encoder(img) if encoder is not None else img
                        output = net(encoded_img)
                        out_fr += output
                        loss += F.cross_entropy(output, label)

            out_fr = out_fr / args.T
            loss /= args.T

            # 反向与优化步（ANN用AMP缩放）
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)
            
            # --- 新增代码：更新进度条 ---
            current_acc = train_acc / train_samples
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.4f}',
                'Batch': f'{batch_idx+1}/{len(train_data_loader)}'
            })
            # --- 新增代码结束 ---

        pbar.close()
        # --- 新增代码结束 ---

        train_time = time.time() - start_time
        train_loss /= train_samples
        train_acc /= train_samples

        train_time_record.append(train_time)

        # --- 新增代码：记录频率权重变化 ---
        if hasattr(net, 'freq_weights') and net.use_freq_weights:
            current_weights = net.freq_weights.detach().cpu().numpy()
            freq_weight_history.append(current_weights)
            
            # 记录权重统计信息
            weight_stats = {
                'epoch': epoch + 1,
                'weights': current_weights,
                'mean': float(np.mean(current_weights)),
                'std': float(np.std(current_weights)),
                'min': float(np.min(current_weights)),
                'max': float(np.max(current_weights)),
                'range': float(np.max(current_weights) - np.min(current_weights))
            }
            
            # 保存当前epoch的权重到PT文件
            weight_filename = f'freq_weights_epoch_{epoch+1:03d}.pt'
            weight_filepath = os.path.join(freq_weight_save_dir, weight_filename)
            torch.save(weight_stats, weight_filepath)
            
            # 记录权重信息到日志
            logger.info(f'频率权重统计 (Epoch {epoch+1}): 均值={weight_stats["mean"]:.4f}, 标准差={weight_stats["std"]:.4f}, 范围=[{weight_stats["min"]:.4f}, {weight_stats["max"]:.4f}]')
            logger.info(f'    前10个权重值: {current_weights[:10].tolist()}')
        # --- 新增代码结束 ---

        # --- 每个epoch后更新学习率 ---
        lr_scheduler.step()

        net.eval()

        start_time = time.time()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        # --- 新增：GPU端每类累计容器 ---
        per_class_counts = torch.zeros(args.num_cls, dtype=torch.long, device=args.device)
        per_class_corrects = torch.zeros(args.num_cls, dtype=torch.long, device=args.device)
        # --- 新增结束 ---

        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(args.device)
                label = label.to(args.device)

                out_fr = 0.
                loss = 0.

                if args.dataset in ['cifar10_dvs', 'nmnist', 'ncaltech']:
                    img = slice_freq(img)
                    img = img.transpose(0, 1)
                    for t in range(args.T):
                        output = net(img[t])
                        out_fr += output
                        loss += F.cross_entropy(output, label)
                else:
                    img = slice_freq(img)
                    for t in range(args.T):
                        encoded_img = encoder(img) if encoder is not None else img
                        output = net(encoded_img)
                        out_fr += output
                        loss += F.cross_entropy(output, label)

                out_fr = out_fr / args.T
                loss /= args.T

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                # 预测与准确累计
                preds = out_fr.argmax(1)
                test_acc += (preds == label).float().sum().item()
                # --- 新增：GPU端每类累计 ---
                per_class_counts += torch.bincount(label, minlength=args.num_cls)
                per_class_corrects += torch.bincount(label[preds == label], minlength=args.num_cls)
                # --- 新增结束 ---
                functional.reset_net(net)

        test_time = time.time() - start_time
        test_loss /= test_samples
        test_acc /= test_samples
        # --- 新增：GPU端计算宏平均与加权（微平均）并写日志 ---
        acc_tensor = per_class_corrects.float() / per_class_counts.clamp_min(1).float()
        macro_acc = acc_tensor.mean().item()
        weighted_acc = (per_class_corrects.sum().float() / per_class_counts.sum().clamp_min(1).float()).item()
        counts_list = per_class_counts.detach().cpu().tolist()
        corrects_list = per_class_corrects.detach().cpu().tolist()
        acc_list = acc_tensor.detach().cpu().tolist()
        logger.info(f'PerClass counts={counts_list} corrects={corrects_list} acc={[float(f"{x:.6f}") for x in acc_list]}')
        # --- 新增结束 ---

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net,
            'max_test_acc': max_test_acc
        }

        # 文件名包含剪枝标签与最佳精度
        _ckpt_name = f'{args.dataset}_{args.model}_{args.act}_T{args.T}'
        if args.freq_keep_idx is not None:
            # 已在前面构造过 _keep_tag；若未导入，可复用 keep_indices 计算
            _total_bins = img_size_ref[args.dataset][0]
            _keep_tag = f'keep{keep_indices.numel()}of{_total_bins}'
            _ckpt_name += f'_{_keep_tag}'
        _ckpt_name += f'_acc{max_test_acc:.4f}.pth'
        torch.save(checkpoint, os.path.join(args.model_dir, _ckpt_name))

        logger.info(f'Epoch[{epoch + 1}/{args.epochs}] train_loss: {train_loss:.4f}, train_acc={train_acc:.4f}, test_loss={test_loss:.4f}, test_acc={test_acc:.4f}, max_test_acc={max_test_acc:.4f}')
        logger.info(f'ACC_Summary: MaxTestAcc={max_test_acc:.6f}, AvgACC={macro_acc:.6f}, WACC={weighted_acc:.6f}')
        logger.info(f'train time: {train_time:.3f}s, test time: {test_time:.3f}s')

    # --- 新增代码：保存完整的权重历史记录 ---
    if hasattr(net, 'freq_weights') and net.use_freq_weights:
        # 保存完整的权重历史
        complete_history = {
            'dataset': args.dataset,
            'architecture': args.model,
            'model_type': args.act,
            'T': args.T,
            'batch_size': args.b,
            'optimizer': args.opt,
            'learning_rate': args.lr,
            'epochs': args.epochs,
            'freq_weight_history': freq_weight_history,
            'final_weights': net.freq_weights.detach().cpu().numpy(),
            'training_completed': True
        }
        
        # 文件名包含精度与剪枝标签
        _hist_name = f'complete_freq_weight_history_{args.dataset}_{args.model}_{args.act}_T{args.T}'
        if args.freq_keep_idx is not None:
            _total_bins = img_size_ref[args.dataset][0]
            _keep_tag = f'keep{keep_indices.numel()}of{_total_bins}'
            _hist_name += f'_{_keep_tag}'
        _hist_name += f'_acc{max_test_acc:.4f}.pt'
        history_filename = _hist_name
        history_filepath = os.path.join(freq_weight_save_dir, history_filename)
        torch.save(complete_history, history_filepath)
        
        logger.info(f'频率权重历史记录已保存到: {history_filepath}')
        logger.info(f'最终频率权重统计: 均值={np.mean(complete_history["final_weights"]):.4f}, 标准差={np.std(complete_history["final_weights"]):.4f}')
    # --- 新增代码结束 ---

    # ================================================================
    # 推理基准测试：延迟 + SynOPs + 能量估计
    # ================================================================
    logger.info('=' * 60)
    logger.info('开始推理基准测试 (Inference Benchmark)')
    logger.info('=' * 60)

    net.eval()
    _bench_device = args.device

    # --- 1. 推理延迟 (Inference Latency) ---
    # 使用 batch_size=1 测量单样本推理时间
    _bench_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    _warmup_iters = 10
    _measure_iters = 50
    _latencies = []

    with torch.no_grad():
        for _i, (_img, _) in enumerate(_bench_loader):
            if _i >= _warmup_iters + _measure_iters:
                break
            _img = _img.to(_bench_device)
            _img = slice_freq(_img)

            if _bench_device != 'cpu':
                torch.cuda.synchronize()
            _t0 = time.time()

            for t in range(args.T):
                _inp = encoder(_img) if encoder is not None else _img
                _ = net(_inp)

            if _bench_device != 'cpu':
                torch.cuda.synchronize()
            _t1 = time.time()

            functional.reset_net(net)

            if _i >= _warmup_iters:
                _latencies.append((_t1 - _t0) * 1000.0)  # ms

    _lat_mean = np.mean(_latencies)
    _lat_std = np.std(_latencies)
    logger.info(f'[Inference Latency] {_lat_mean:.3f} ± {_lat_std:.3f} ms/sample  (T={args.T}, device={_bench_device}, measured over {len(_latencies)} samples)')

    # --- 2. Spike Count & SynOPs ---
    from utils import spike_count_per_layer

    _synops_samples = min(100, len(test_dataset))
    _synops_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0)

    _agg_total_spikes = 0
    _agg_total_synops = 0
    _agg_layer_spikes = {}
    _agg_layer_synops = {}
    _agg_count = 0

    with torch.no_grad():
        for _i, (_img, _) in enumerate(_synops_loader):
            if _i >= _synops_samples:
                break
            _img = _img.to(_bench_device)
            _img = slice_freq(_img)

            _ds_type = args.dataset if args.dataset in ['cifar10_dvs', 'nmnist', 'ncaltech'] else 'static'
            _result = spike_count_per_layer(net, _img, args.T, encoder=encoder, dataset_type=_ds_type)

            _agg_total_spikes += _result['total_spikes']
            _agg_total_synops += _result['total_synops']
            for _ls in _result['layer_stats']:
                _agg_layer_spikes.setdefault(_ls['name'], 0)
                _agg_layer_spikes[_ls['name']] += _ls['spikes']
                _agg_layer_synops.setdefault(_ls['name'], 0)
                _agg_layer_synops[_ls['name']] += _ls['synops']
            _agg_count += 1

    _avg_spikes = _agg_total_spikes / max(_agg_count, 1)
    _avg_synops = _agg_total_synops / max(_agg_count, 1)

    logger.info(f'[Spike Count] Avg total spikes per sample: {_avg_spikes:.0f}  (over {_agg_count} samples)')
    logger.info(f'[SynOPs]      Avg total SynOPs per sample: {_avg_synops:.0f}')
    logger.info(f'[Per-Layer Breakdown]')
    for _lname in sorted(_agg_layer_spikes.keys(), key=lambda x: int(x.split('_')[1])):
        _ls = _agg_layer_spikes[_lname] / max(_agg_count, 1)
        _lo = _agg_layer_synops[_lname] / max(_agg_count, 1)
        logger.info(f'    {_lname}: avg_spikes={_ls:.0f}, avg_synops={_lo:.0f}')

    # --- 3. 45nm 能量估计 ---
    E_MAC = 4.6   # pJ, 32-bit FP MAC @ 45nm
    E_AC  = 0.9   # pJ, 32-bit AC  @ 45nm
    _energy_ac = _avg_synops * E_AC       # SNN 突触操作以 AC 为主
    _energy_mac = 0.0                     # 首层频率加权为 MAC 操作
    if args.use_freq_weights or (keep_indices is not None):
        _orig_h = img_size_ref[args.dataset][0]
        _actual_h = keep_indices.numel() if keep_indices is not None else _orig_h
        _w_dim = img_size_ref[args.dataset][1]
        _energy_mac = _actual_h * _w_dim * args.T * E_MAC
    _energy_total = _energy_ac + _energy_mac

    logger.info(f'[Energy Est. 45nm] AC_energy={_energy_ac:.1f} pJ, MAC_energy={_energy_mac:.1f} pJ, Total={_energy_total:.1f} pJ  (per sample, T={args.T})')

    # --- 汇总指标 ---
    logger.info('=' * 60)
    logger.info('Metrics Summary:')
    logger.info(f'  Accuracy:          {max_test_acc:.4f}')
    logger.info(f'  Train time/epoch:  {np.mean(train_time_record):.4f}s')
    logger.info(f'  Infer latency:     {_lat_mean:.3f} ± {_lat_std:.3f} ms/sample')
    logger.info(f'  Avg spikes/sample: {_avg_spikes:.0f}')
    logger.info(f'  Avg SynOPs/sample: {_avg_synops:.0f}')
    logger.info(f'  Energy (45nm):     {_energy_total:.1f} pJ/sample')
    logger.info('=' * 60)


if args.prune:
    ensure_dir(os.path.join(args.split_dir, f'{args.dataset}_{args.model}_{args.act}_T{args.T}_checkpoint'))
    
    checkpoint = torch.load(os.path.join(args.model_dir, f'{args.dataset}_{args.model}_{args.act}_T{args.T}_checkpoint_max.pth'), map_location='cpu')
    net = checkpoint['net']
    logger.info(f'Load existing model')

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=len(train_dataset),
        shuffle=True,
        drop_last=True, 
        num_workers=args.j, 
        pin_memory=True)

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=len(test_dataset),
        shuffle=False,
        drop_last=False, 
        num_workers=args.j, 
        pin_memory=True)
    
    # select target class
    target_class = sorted(args.split_class) 
    pmodel_name = ''.join([str(x) for x in target_class])

    p_num_cls = len(target_class) + 1

    x_train, y_train = next(iter(train_data_loader))
    x_test, y_test = next(iter(test_data_loader))
    del train_dataset, test_dataset

    x_retrain_train, y_retrain_train , x_retrain_test, y_retrain_test = split_retrain(
        x_train, x_test , y_train, y_test, target_class, args.num_cls)
    logger.info('Finish extracting pruning data from original data')

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_dataset = CustomDataset(x_retrain_train, y_retrain_train)
    test_dataset = CustomDataset(x_retrain_test, y_retrain_test)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.b,
        shuffle=True,
        drop_last=True, 
        num_workers=args.j, 
        pin_memory=True)

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.b,
        shuffle=False,
        drop_last=False, 
        num_workers=args.j, 
        pin_memory=True)

    prune_sample = x_retrain_train[y_retrain_train!=0] # (N,C,H,W) for non-neu data, (N, T, C, H, W) for neu data
    logger.info(f'pruned samples number: {prune_sample.shape}')

    iter_len = args.b # set chunck size same as batch size for train, then it will always meet the memory limitation
    idx_set = list(range(0, len(prune_sample), iter_len))
    if idx_set[-1] + iter_len >= len(prune_sample):
        idx_set.append(len(prune_sample)) 

    activation = {}
    def get_activation(name):
        def hook(m, input, output):
            activation[name] = output.detach()
        return hook
    
    follow_conv = False
    num_conv = 0 
    i_h_dim = img_size_ref[args.dataset][0]
    i_w_dim = img_size_ref[args.dataset][1]
    area_conv_hw = []
    for n, l in net.conv_fc.named_children():
        if isinstance(l, layer.Conv2d):
            follow_conv = True
            hw = i_h_dim * i_w_dim
            area_conv_hw.append(hw)
        elif isinstance(l, (neuron.LIFNode, nn.ReLU)) and follow_conv is True:
            l.register_forward_hook(get_activation(f'conv{num_conv}'))
            num_conv += 1
            follow_conv = False
            print(f'conv with index {n} in nn.sequential is hooked')
        elif isinstance(l, (layer.AvgPool2d, layer.MaxPool2d)):
            i_h_dim //= 2
            i_w_dim //= 2
        else:
            pass
    logger.info(f'Finish add forward-hook to model, total conv layer number: {num_conv}') 

    areas = np.array(area_conv_hw)
    apozs = [0. for _ in range(num_conv)]

    net.to(args.device)

    p_cnt = 0
    for start_idx, end_idx in zip(idx_set[:-1], idx_set[1:]):
        p_prune_sample = prune_sample[start_idx:end_idx] # N, C, H, W / N, T, C, H, W
        p_prune_sample = p_prune_sample.to(args.device)

        nzs = [0. for _ in range(num_conv)]

        if args.dataset in ['cifar10_dvs', 'nmnist', 'ncaltech']:
            p_prune_sample = p_prune_sample.transpose(0, 1)
            for t in range(args.T):
                _ = net(p_prune_sample[t])
                for k in range(num_conv):
                    act = torch.permute(activation[f'conv{k}'], dims=[1,0,2,3]) # N,C,H,W -> C, N, H, W
                    act = act.reshape(act.shape[0], -1) # C, N, H, W -> C, NHW
                    nz = act.shape[1] - act.count_nonzero(dim=1) # find out how many zero in NHW 
                    nzs[k] += nz
        else:
            for t in range(args.T):
                encoded_img = encoder(p_prune_sample) if encoder is not None else p_prune_sample
                _ = net(encoded_img)
                for k in range(num_conv):
                    act = torch.permute(activation[f'conv{k}'], dims=[1,0,2,3]) # N,C,H,W -> C, N, H, W
                    act = act.reshape(act.shape[0], -1) # C, N, H, W -> C, NHW
                    nz = act.shape[1] - act.count_nonzero(dim=1) # find out how many zero in NHW 
                    nzs[k] += nz

        nzs = [nz / args.T for nz in nzs]  # non-zero number per conv per T
        p_areas = areas * (end_idx - start_idx) # Nhw
        
        for nc in range(num_conv):
            apoz = nzs[nc] / p_areas[nc]
            apozs[nc] += apoz
        
        print(f'Finish extracting apoz with pruned samples {start_idx}-{end_idx}')
        p_cnt += 1

        functional.reset_net(net)

    apozs = [(apoz / p_cnt).tolist() for apoz in apozs]

    # print(apozs)

    layer_index = []
    for nc in range(num_conv):
        apoz = apozs[nc]
        idxs = []
        for na in range(len(apoz)):
            A = apoz[na]
            if (A > ((args.apoz_threshold - nc) / 100)) & (args.min_f < len(apoz) - idxs.count(0)):
                idxs.append(0)
            else:
                idxs.append(1)
        layer_index.append(idxs)

    logger.info('Finish ranking filter importance')
    for i in range(len(layer_index)):
        ele = layer_index[i]
        logger.info(f'conv{i} shrink to {np.sum(ele)}')

    new_conv_fc = nn.Sequential()
    c_dim = input_dim_ref[args.dataset]
    i_h_dim = img_size_ref[args.dataset][0]
    i_w_dim = img_size_ref[args.dataset][1]
    bias_flag = True if args.act == 'ann' else False
    index = 0
    conv_idx_rec = []
    conv_idx = 0

    for l in net.conv_fc:
        if isinstance(l, layer.Conv2d):
            channels = np.array(layer_index[index])
            num_channels = channels.sum()
            new_conv_fc.append(
                layer.Conv2d(c_dim, num_channels, kernel_size=3, padding=1, bias=bias_flag))

            c_dim = num_channels
            index += 1
            conv_idx_rec.append(conv_idx)
        elif isinstance(l, layer.BatchNorm2d):
            new_conv_fc.append(layer.BatchNorm2d(c_dim)) 
        elif isinstance(l, (layer.AvgPool2d, layer.MaxPool2d)):
            new_conv_fc.append(
                layer.AvgPool2d(kernel_size=2) if args.act == 'snn' else layer.MaxPool2d(kernel_size=2))
            i_h_dim //= 2
            i_w_dim //= 2
        elif isinstance(l, layer.Dropout):
            new_conv_fc.append(layer.Dropout(0.5))
        elif isinstance(l, layer.Flatten):
            new_conv_fc.append(layer.Flatten())
        elif isinstance(l, nn.Linear):
            new_conv_fc.append(layer.Linear(i_h_dim * i_w_dim * c_dim, 384, bias=bias_flag))
            new_conv_fc.append(layer.Dropout(0.5))
            new_conv_fc.append(layer.Linear(384, 192, bias=bias_flag))
            new_conv_fc.append(layer.Dropout(0.5))
            new_conv_fc.append(layer.Linear(192, p_num_cls, bias=bias_flag))
            break

        elif isinstance(l, nn.ReLU):
            new_conv_fc.append(nn.ReLU())
        elif isinstance(l, neuron.LIFNode):
            new_conv_fc.append(
                neuron.LIFNode(tau=args.tau, v_reset=None, surrogate_function=surrogate.ATan(), detach_reset=True))
        else:
            raise NotImplementedError(f'Unknown Layer name: {l}')
        conv_idx += 1
            
    logger.info('pruned model architecture: \n' + str(new_conv_fc) + '\nFinish build pruned model architecture')

    old_weights = [net.conv_fc[conv_idx].weight.data for conv_idx in conv_idx_rec]

    for idx, conv_idx in enumerate(conv_idx_rec):

        current_old_weights = old_weights[idx] # C_out, C_in, H, W
        preserve_index = torch.tensor(layer_index[idx])
        new_weights = current_old_weights[preserve_index == 1]  # new_C_out, C_in, H, W
        new_conv_fc[conv_idx].weight = nn.Parameter(new_weights) # assign to pre-conv

        if idx < len(conv_idx_rec) - 1:
            next_weights = torch.permute(old_weights[idx + 1], dims=[1,0,2,3]) # C_out, C_in, H, W -> C_in, C_out, H, W
            next_weights = next_weights[preserve_index == 1] # new_C_in, C_out, H, W
            next_weights = torch.permute(next_weights, dims=[1, 0, 2, 3]) #  new_C_in, C_out, H, W -> C_out, new_C_in, H, W
            old_weights[idx + 1] = next_weights
        
    logger.info('Finish loading pretrained weight to pruned model')

    net.conv_fc = new_conv_fc
    net.to(args.device)

    logger.info('\n' + str(net))
    for l in net.conv_fc:
        if isinstance(l, (layer.Conv2d, layer.Linear)):
            print(f'{l.weight.data.shape}')

    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(f'invalid optimizer: {args.opt}')

    # --- 新增学习率调度器 ---
    if args.scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    else: # 默认为阶梯式衰减
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    print('Start retraining...')
    max_test_acc = -1
    for epoch in range(args.epochs):
        if epoch + 1 > 20:
            break
        print(f'start epoch {epoch + 1}')
        net.train()

        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in train_data_loader:
            optimizer.zero_grad()
            img = img.to(args.device)
            label = label.to(args.device)

            out_fr = 0.
            loss = 0.

            if args.dataset in ['cifar10_dvs', 'nmnist', 'ncaltech']:
                img = img.transpose(0, 1)
                for t in range(args.T):
                    output = net(img[t])
                    out_fr += output
                    loss += F.cross_entropy(output, label)
            else:
                for t in range(args.T):
                    encoded_img = encoder(img) if encoder is not None else img
                    output = net(encoded_img)
                    out_fr += output
                    loss += F.cross_entropy(output, label)

            out_fr = out_fr / args.T
            loss /= args.T

            # --- 应用 Mixup ---
            if args.mixup and args.dataset == 'esc50': # 只对ESC50应用
                inputs, targets_a, targets_b, lam = mixup_data(img, label, args.mixup_alpha, True)
                outputs = net(inputs)
                loss = mixup_criterion(F.cross_entropy, outputs, targets_a, targets_b, lam)
            else:
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        train_loss /= train_samples
        train_acc /= train_samples

        lr_scheduler.step()

        net.eval()

        test_loss = 0
        test_acc = 0
        test_samples = 0

        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(args.device)
                label = label.to(args.device)

                out_fr = 0.
                loss = 0.

                if args.dataset in ['cifar10_dvs', 'nmnist', 'ncaltech']:
                    img = img.transpose(0, 1)
                    for t in range(args.T):
                        output = net(img[t])
                        out_fr += output
                        loss += F.cross_entropy(output, label)
                else:
                    for t in range(args.T):
                        encoded_img = encoder(img) if encoder is not None else img
                        output = net(encoded_img)
                        out_fr += output
                        loss += F.cross_entropy(output, label)

                out_fr = out_fr / args.T
                loss /= args.T

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)

        test_loss /= test_samples
        test_acc /= test_samples

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net, 
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(
                checkpoint, 
                os.path.join(args.split_dir, f'{args.dataset}_{args.model}_{args.act}_T{args.T}_checkpoint', f'{pmodel_name}.pth'))
            
        logger.info(f'Epoch[{epoch + 1}/{args.epochs}] train_loss: {train_loss:.4f}, train_acc={train_acc:.4f}, test_loss={test_loss:.4f}, test_acc={test_acc:.4f}, max_test_acc={max_test_acc:.4f}')

    logger.info('Finish retrain the pruned model')


if args.fusion:
    logger.info('Start fusion model building...')

    splitted_model_dir = os.path.join(args.split_dir, f'{args.dataset}_{args.model}_{args.act}_T{args.T}_checkpoint/')
    model_names = [f for f in os.listdir(splitted_model_dir) if f.endswith('.pth')]
    pmodels = []
    for model_name in model_names:
        logger.info(f'Load {model_name} ')
        pth = torch.load(os.path.join(splitted_model_dir, model_name), map_location='cpu')
        pmodels.append(pth['net'])

    in_dim = np.sum([pmodel.conv_fc[-5].in_features for pmodel in pmodels]) # the first linear layer in feature
    pmodels = [pmodel.conv_fc[:-5] for pmodel in pmodels] # each pruned model end before the first linear layer appear
    pmodel_num = len(model_names)
        
    fusion_model = Fusion(in_dim, pmodel_num, args.num_cls, args.act)
    fusion_model.to(args.device)

    for pmodel in pmodels:
        for param in pmodel.parameters():
            param.requires_grad = False

    pmodels = [pmodel.to(args.device) for pmodel in pmodels]

    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(fusion_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(fusion_model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(f'invalid optimizer: {args.opt}')
    
    # --- 新增学习率调度器 ---
    if args.scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    else: # 默认为阶梯式衰减
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    max_test_acc = -1
    for epoch in range(args.epochs):
        if epoch + 1 > 15:
            break
        print(f'start epoch {epoch + 1}')
        fusion_model.train()

        train_loss = 0
        train_acc = 0
        train_samples = 0
        start_time = time.time()
        for img, label in train_data_loader:
            optimizer.zero_grad()
            img = img.to(args.device)
            label = label.to(args.device)

            out_fr = 0.
            loss = 0.

            if args.dataset in ['cifar10_dvs', 'nmnist', 'ncaltech']:
                # for neuro dataset, dataloader generate (N, T, C, H, W), so we need change (T, N, C, H, W)
                img = img.transpose(0, 1)
                for t in range(args.T):
                    fes = torch.concat([pmodel(img[t]) for pmodel in pmodels], dim=1)
                    output = fusion_model(fes)
                    out_fr += output
                    loss += F.cross_entropy(output, label)
            else:
                for t in range(args.T):
                    encoded_img = encoder(img) if encoder is not None else img
                    fes = torch.concat([pmodel(encoded_img) for pmodel in pmodels], dim=1)
                    output = fusion_model(fes)
                    out_fr += output
                    loss += F.cross_entropy(output, label)

            out_fr = out_fr / args.T
            loss /= args.T

            # --- 应用 Mixup ---
            if args.mixup and args.dataset == 'esc50': # 只对ESC50应用
                inputs, targets_a, targets_b, lam = mixup_data(img, label, args.mixup_alpha, True)
                outputs = fusion_model(inputs)
                loss = mixup_criterion(F.cross_entropy, outputs, targets_a, targets_b, lam)
            else:
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(fusion_model)
            for pmodel in pmodels:
                functional.reset_net(pmodel)

        train_time = time.time() - start_time
        train_loss /= train_samples
        train_acc /= train_samples

        lr_scheduler.step()

        fusion_model.eval()

        test_loss = 0
        test_acc = 0
        test_samples = 0
        start_time = time.time()

        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(args.device)
                label = label.to(args.device)

                out_fr = 0.
                loss = 0.

                if args.dataset in ['cifar10_dvs', 'nmnist', 'ncaltech']:
                    img = img.transpose(0, 1)
                    for t in range(args.T):
                        fes = torch.concat([pmodel(img[t]) for pmodel in pmodels], dim=1)
                        output = fusion_model(fes)  
                        out_fr += output
                        loss += F.cross_entropy(output, label)
                else:
                    for t in range(args.T):
                        encoded_img = encoder(img) if encoder is not None else img
                        fes = torch.concat([pmodel(encoded_img) for pmodel in pmodels], dim=1)
                        output = fusion_model(fes)  
                        out_fr += output
                        loss += F.cross_entropy(output, label)

                out_fr = out_fr / args.T
                loss /= args.T

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(fusion_model)
                for pmodel in pmodels:
                    functional.reset_net(pmodel)
        
        test_loss /= test_samples
        test_acc /= test_samples
        test_time = time.time() - start_time

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': fusion_model,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(fusion_model, f'{splitted_model_dir}/fusion.pt')

        logger.info(f'Epoch[{epoch + 1}/{args.epochs}] train_loss: {train_loss:.4f}, train_acc={train_acc:.4f}, test_loss={test_loss:.4f}, test_acc={test_acc:.4f}, max_test_acc={max_test_acc:.4f}')
        logger.info(f'train time: {train_time:.3f}s, test time: {test_time:.3f}s')

    logger.info(f'Max Accuracy for Fusion Model: {max_test_acc:.4f}')
    

if args.infer:
    # infer must be tested on cpu since it will further set to raspberry

    if not os.path.exists('./infer_data/'):
        raise NotImplementedError('extract infer data by yourself first...')

    logger.info('Load data for inference')
    if args.dataset == 'mnist':
        img = torch.load('./infer_data/mnist_frame.pt', map_location='cpu')
    elif args.dataset == 'cifar10':
        img = torch.load('./infer_data/cifar10_frame.pt', map_location='cpu')
    elif args.dataset == 'caltech':
        img = torch.load('./infer_data/caltech_frame.pt', map_location='cpu')
    elif args.dataset == 'cifar10_dvs':
        img = torch.load('./infer_data/cifar10_dvs_frame.pt', map_location='cpu')
    elif args.dataset == 'nmnist':
        img = torch.load('./infer_data/nmnist_frame.pt', map_location='cpu')
    elif args.dataset == 'ncaltech':
        img = torch.load('./infer_data/ncaltech_frame.pt', map_location='cpu')
    elif args.dataset == 'gtzan':
        img = torch.load('./infer_data/gtzan_frame.pt', map_location='cpu')
    elif args.dataset == 'urbansound':
        img = torch.load('./infer_data/urbansound_frame.pt', map_location='cpu')
    elif args.dataset == 'esc50':
        img = torch.load('./infer_data/esc50_frame.pt', map_location='cpu')
    else:
        raise NotImplementedError(f'Invalid dataset name: {args.dataset}...')
    
    if args.split:
        # EC split infer mode
        splitted_model_dir = os.path.join(args.split_dir, f'{args.dataset}_{args.model}_{args.act}_T{args.T}_checkpoint/')
        pmodels = []
        model_names = [f for f in os.listdir(splitted_model_dir) if f.endswith('.pth')]
        fusion_name = [f for f in os.listdir(splitted_model_dir) if f.endswith('.pt')][0]

        for model_name in model_names:
            logger.info(f'Load {model_name} ')
            pth = torch.load(os.path.join(splitted_model_dir, model_name), map_location='cpu')
            pmodels.append(pth['net'])


        pmodels = [pmodel.conv_fc[:-5] for pmodel in pmodels]
        fusion_model = torch.load(os.path.join(splitted_model_dir, fusion_name), map_location='cpu')
        
        pmodels = [pmodel.to(args.device) for pmodel in pmodels]
        fusion_model.to(args.device)
        img = img.unsqueeze(0).to(args.device)

        pmodels = [pmodel.eval() for pmodel in pmodels]
        fusion_model.eval()

        with torch.no_grad():
            out_fr = 0.
            infer_time_record = []
            pmodels_infer_time_record = []  # (T, # of pmodel)

            if args.dataset in ['cifar10_dvs', 'nmnist', 'ncaltech']:
                img = img.transpose(0, 1)
                for t in range(args.T):

                    # store the infer time of each model at time t
                    fe_time_rec = []
                    fe_rec = []
                    for pmodel in pmodels:
                        start_time = time.time()
                        out = pmodel(img[t])
                        fe_time_rec.append(time.time() - start_time)
                        fe_rec.append(out)

                    fe = torch.concat(fe_rec, dim=1)

                    elapse_time = max(fe_time_rec)

                    # store all the infer time of each model at all T time
                    pmodels_infer_time_record.append(fe_time_rec)

                    start_time = time.time()
                    out = fusion_model(fe)
                    elapse_time += (time.time() - start_time)
                    out_fr += out
                    infer_time_record.append(elapse_time)
            else:
                for t in range(args.T):
                    encoded_img = encoder(img) if encoder is not None else img

                    fe_time_rec = []
                    fe_rec = []

                    for pmodel in pmodels:
                        start_time = time.time()
                        out = pmodel(encoded_img)
                        fe_time_rec.append(time.time() - start_time)
                        fe_rec.append(out)

                    fe = torch.concat(fe_rec, dim=1)
                    elapse_time = max(fe_time_rec)
                    pmodels_infer_time_record.append(fe_time_rec)

                    start_time = time.time()
                    out = fusion_model(fe)
                    elapse_time += (time.time() - start_time)
                    out_fr += out
                    infer_time_record.append(elapse_time)

            logger.info(f'infer time: {np.mean(infer_time_record):.4f}s')
            pmodels_infer_time_record = np.array(pmodels_infer_time_record)
            # T*device_num records to 1*device_num，represent avg time for pmodel infer one frame
            res = pmodels_infer_time_record.mean(axis=0)
            cankao = [f'{k}: {v:.4f}' for k, v in zip(model_names, res)]
            logger.info(f'average of infer one frame of a sample per pruned model: {np.mean(res):.4f}, max: {np.max(res):.4f}, min:{np.min(res):.4f}, all: {cankao}') 


    else:
        # single mode
        if os.path.exists(os.path.join(args.model_dir, f'{args.dataset}_{args.model}_{args.act}_T{args.T}_checkpoint_max.pth')):
            checkpoint = torch.load(os.path.join(args.model_dir, f'{args.dataset}_{args.model}_{args.act}_T{args.T}_checkpoint_max.pth'), map_location='cpu')
            net = checkpoint['net']
            logger.info(f'Load existing model')

        net.to(args.device)
        img = img.unsqueeze(0).to(args.device) 

        net.eval() #
        with torch.no_grad():
            out_fr = 0.
            infer_time_record = []
            if args.dataset in ['cifar10_dvs', 'nmnist', 'ncaltech']:
                img = img.transpose(0, 1)
                for t in range(args.T):
                    start_time = time.time()
                    out_fr += net(img[t])
                    infer_time_record.append(time.time() - start_time)
            else:
                for t in range(args.T):
                    encoded_img = encoder(img) if encoder is not None else img
                    start_time = time.time()
                    out_fr += net(encoded_img)
                    infer_time_record.append(time.time() - start_time)

            print(f'infer time: {np.mean(infer_time_record):.4f}s')


if args.energy:
    if False:
        del train_dataset, train_data_loader

        if args.split:
            pmodels = []
            splitted_model_dir = os.path.join(args.split_dir, f'{args.dataset}_{args.model}_{args.act}_T{args.T}_checkpoint/')
            model_names = [f for f in os.listdir(splitted_model_dir) if f.endswith('.pth')]
            fusion_name = [f for f in os.listdir(splitted_model_dir) if f.endswith('.pt')][0]

            for model_name in model_names:
                logger.info(f'Load {model_name} ')
                pth = torch.load(os.path.join(splitted_model_dir, model_name), map_location='cpu')
                pmodels.append(pth['net'])

            pmodels = [pmodel.conv_fc[:-5] for pmodel in pmodels]
            fusion_model = torch.load(os.path.join(splitted_model_dir, fusion_name), map_location='cpu')

            pmodels = [pmodel.to(args.device) for pmodel in pmodels]
            fusion_model.to(args.device)

            device_num = len(pmodels)

            with torch.no_grad():
                rec = []
                spike_per_model_rec = []

                for img, _ in test_data_loader:
                    img = img.to(args.device)

                    spike_num = 0.
                    b = img.shape[0]
                    spike_per_model = [0 for _ in range(device_num)]

                    if args.dataset in ['cifar10_dvs', 'nmnist', 'ncaltech']:
                        img = img.transpose(0, 1)
                        for t in range(args.T):
                            fe_rec = []
                            # for pmodel in pmodels:
                            for d in range(device_num):
                                pmodel = pmodels[d]

                                cnt, out = spike_count(pmodel, img[t])

                                spike_per_model[d] += cnt

                                spike_num += cnt
                                fe_rec.append(out)

                            fe = torch.concat(fe_rec, dim=1)
                            cnt, _ = spike_count(fusion_model.mlp, fe)
                            spike_num += cnt

                    else:
                        for t in range(args.T):
                            encoded_img = encoder(img) if encoder is not None else img
                            fe_rec = []
                            # for pmodel in pmodels:
                            for d in range(device_num):
                                pmodel = pmodels[d]

                                cnt, out = spike_count(pmodel, encoded_img)

                                #  store spike numbers for each pmodel infer one batch for T times
                                spike_per_model[d] += cnt

                                spike_num += cnt
                                fe_rec.append(out)

                            fe = torch.concat(fe_rec, dim=1)
                            cnt, _ = spike_count(fusion_model.mlp, fe)
                            # double-check, no spike in FC
                            assert cnt == 0, 'you know'
                            spike_num += cnt

                    functional.reset_net(fusion_model)

                    for pmodel in pmodels:
                        functional.reset_net(pmodel)

                    res = spike_num / b
                    rec.append(res.cpu())

                    # avg spikes for avg infer one sample in T times
                    spike_per_model = [num / b for num in spike_per_model]
                    spike_per_model_rec.append(spike_per_model)

            spike_per_model_rec = torch.tensor(spike_per_model_rec).cpu().numpy()  # batch_num * device_num
            # min max mean for pmodels, so need to convert batch_num * device_num to 1 * device_num
            spike_per_model_rec = spike_per_model_rec.mean(axis=0)

            cankao = {f'{k}: {v:2f}' for k, v in zip(model_names, spike_per_model_rec.tolist())}

            logger.info(
                f'average spikes for one test sample with {args.T} frames in {args.dataset} with architecture {args.act}-{args.model}: {np.sum(spike_per_model_rec) / device_num:.2f}, std: {np.std(spike_per_model_rec):.2f}, min: {np.min(spike_per_model_rec):.2f}, max: {np.max(spike_per_model_rec):.2f}, details per model: {cankao}')

        else:
            if os.path.exists(
                    os.path.join(args.model_dir, f'{args.dataset}_{args.model}_{args.act}_T{args.T}_checkpoint_max.pth')):
                checkpoint = torch.load(
                    os.path.join(args.model_dir, f'{args.dataset}_{args.model}_{args.act}_T{args.T}_checkpoint_max.pth'),
                    map_location='cpu')
                net = checkpoint['net']
                print(f'Load existing model')

            net.to(args.device)
            net.eval()

            with torch.no_grad():
                rec = []
                for img, _ in test_data_loader:
                    spike_num = 0.
                    img = img.to(args.device)
                    b = img.shape[0]

                    if args.dataset in ['cifar10_dvs', 'nmnist', 'ncaltech']:
                        img = img.transpose(0, 1)
                        for t in range(args.T):
                            cnt, _ = spike_count(pmodel, img[t], img[t].count_nonzero() / img[t].numel())
                            spike_num += cnt

                    else:
                        for t in range(args.T):
                            encoded_img = encoder(img) if encoder is not None else img
                            cnt, _ = spike_count(net.conv_fc, encoded_img)
                            spike_num += cnt

                    functional.reset_net(net)

                    res = spike_num / b
                    rec.append(res.cpu())

            logger.info(f'average spikes for one test samples with {args.T} frames in {args.dataset}')