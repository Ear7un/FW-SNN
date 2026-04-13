import os
import cv2
import librosa
import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio import transforms as T

import torchvision
from torchvision import transforms

from spikingjelly.activation_based import neuron
from spikingjelly.datasets.n_mnist import NMNIST
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.n_caltech101 import NCaltech101 


dataset = ['mnist', 'cifar10', 'caltech', 'cifar10_dvs', 'nmnist', 'ncaltech']

img_size_ref = {
    'mnist': (28, 28),
    'nmnist': (34, 34),
    'cifar10': (32, 32),
    'cifar10_dvs': (128, 128),
    'caltech': (224, 224),
    'ncaltech': (224, 224),
    'gtzan': (96, 323),
    'urbansound': (40, 173), 
    'esc50': (128, 216), # 恢复使用 MelSpectrogram 对应的尺寸
}

input_dim_ref = {
    'mnist' : 1,
    'nmnist': 2,
    'cifar10': 3,
    'cifar10_dvs': 2,
    'caltech': 3,
    'ncaltech': 2,
    'gtzan': 1,
    'urbansound': 1,
    'esc50': 1,
}

# --- 移除所有之前为ESC-50添加的、不再需要的自定义类和函数 ---

# --- 恢复之前为ESC-50设计的、效果更好的数据处理流程 ---
class PowerToDB_ref_max(nn.Module):
    def __init__(self, top_db=80.0):
        super().__init__()
        self.top_db = top_db

    def forward(self, spec):
        db_spec = T.AmplitudeToDB(stype='power', top_db=self.top_db)(spec)
        # 仿效 librosa.power_to_db(ref=np.max)
        return db_spec - db_spec.max()

class ESC50OptimizedDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, audio_dir, transform=None, target_sr=22050, duration=5):
        self.audio_labels = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transform = transform
        self.target_sr = target_sr
        self.num_samples = int(target_sr * duration)

    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, index):
        audio_sample_path = os.path.join(self.audio_dir, 'audio', 'audio', self.audio_labels.iloc[index, 0])
        label = self.audio_labels.iloc[index, 2]
        try:
            signal, sr = torchaudio.load(audio_sample_path)
            if sr != self.target_sr:
                signal = torchaudio.functional.resample(signal, sr, self.target_sr)

            if signal.shape[1] > self.num_samples:
                signal = signal[:, :self.num_samples]
            else:
                signal = F.pad(signal, (0, self.num_samples - signal.shape[1]))
        except Exception as e:
            print(f"Error loading or processing file {audio_sample_path}: {e}")
            signal = torch.zeros(1, self.num_samples)
            label = 0
        
        if self.transform:
            signal = self.transform(signal)
            
        return signal, label

def get_esc50_optimized_dataset(path='./ESC-50/', n_mels=128, duration=5):
    train_transform = nn.Sequential(
        T.MelSpectrogram(sample_rate=22050, n_mels=n_mels, n_fft=1024, hop_length=512),
        PowerToDB_ref_max(),
        T.FrequencyMasking(freq_mask_param=25),
        T.TimeMasking(time_mask_param=50)
    )
    test_transform = nn.Sequential(
        T.MelSpectrogram(sample_rate=22050, n_mels=n_mels, n_fft=1024, hop_length=512),
        PowerToDB_ref_max()
    )

    dataset = ESC50OptimizedDataset(os.path.join(path, "esc50.csv"), path, duration=duration)
    test_indices = dataset.audio_labels['fold'] == 5
    
    train_dataset = torch.utils.data.Subset(dataset, np.where(~test_indices)[0])
    test_dataset = torch.utils.data.Subset(dataset, np.where(test_indices)[0])
    
    # 动态地将不同的变换应用到数据子集
    train_dataset.dataset.transform = train_transform
    test_dataset.dataset.transform = test_transform
    
    return train_dataset, test_dataset


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')

    return cur

class Logger:
    ''' spikingjelly induce logging not work '''
    def __init__(self, args, state=None, desc=None):
        # 修改：直接使用args.out_dir作为日志文件位置
        log_root = args.out_dir
        ensure_dir(log_root)

        # 修改：直接在out_dir下创建日志文件，不再创建子目录
        logfilename = f'{desc}_record.log'
        logfilepath = os.path.join(log_root, logfilename)

        self.filename = logfilepath

        f = open(logfilepath, 'w', encoding='utf-8')
        f.write(str(args) + '\n')
        f.flush()
        f.close()

    def info(self, s=None):
        print(s)
        f = open(self.filename, 'a', encoding='utf-8')
        f.write(f'[{get_local_time()}] - {s}\n')
        f.flush()
        f.close()



def generate_infer_one_sample(T=5):
    print('Load infer data')
    if not os.path.exists('./infer_data/'):
        os.makedirs('./infer_data/')

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = torchvision.datasets.MNIST(
        root='.',
        train=False,
        transform=transform_test,
        download=True)
    frame, _ = next(iter(test_dataset))
    print('mnist: ', frame.shape, type(frame))
    torch.save(frame, './infer_data/mnist_frame.pt')

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    test_dataset = torchvision.datasets.CIFAR10(
        root='.',
        train=False,
        transform=transform_test,
        download=True)  
    frame, _ = next(iter(test_dataset))
    print('cifar10: ', frame.shape, type(frame))
    torch.save(frame, './infer_data/cifar10_frame.pt')

    transform_all = transforms.Compose([
        # transforms.ToPILImage(),  # already PILImage
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3,1,1) if x.shape[0] == 1 else x),
        transforms.Normalize(mean = [0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    test_dataset = torchvision.datasets.Caltech101(
        root='.',
        transform=transform_all,
        download=True)

    frame, _ = next(iter(test_dataset))
    print('caltech: ', frame.shape, type(frame))
    torch.save(frame, './infer_data/caltech_frame.pt')

    test_dataset = CIFAR10DVS(
        root='./cifar10_dvs', 
        data_type='frame', 
        frames_number=T, 
        split_by='number')

    frame, _ = next(iter(test_dataset))
    frame = torch.tensor(frame)
    print('cifar10-dvs: ', frame.shape, type(frame))
    torch.save(frame, './infer_data/cifar10_dvs_frame.pt')

    test_dataset = NMNIST(
        root='./NMNIST',
        train=False,
        data_type='frame',
        frames_number=T,
        split_by='number')

    frame, _ = next(iter(test_dataset))
    frame = torch.tensor(frame)
    print('n-mnist: ', frame.shape, type(frame))
    torch.save(frame, './infer_data/nmnist_frame.pt')

    transform_all = transforms.Compose([
        transforms.Lambda(lambda x: torch.tensor(x)),
        transforms.Resize((224, 224), antialias=True),
    ])

    test_dataset = NCaltech101(
        root='./NCaltech', 
        data_type='frame',
        frames_number=T,
        transform=transform_all,
        split_by='number')

    frame, _ = next(iter(test_dataset))
    print('n-caltech: ', frame.shape, type(frame))
    torch.save(frame, './infer_data/ncaltech_frame.pt')

# select top10 from 101
class CaltechTop10(torch.utils.data.Dataset):
    def __init__(self, data):
        top10 = [5, 3, 0, 1, 94, 2, 12, 19, 55, 23]
        ref = {top10[i] : i for i in range(10)}
        self.data = []
        for x, y in data:
            if y in top10:
                self.data.append((x, ref[y]))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tar = self.data[idx]
        return tar[0], tar[1]
    
def split_retrain(x_train, x_test, y_train, y_test, target_class, num_cls=10):
    ''' resample dataset for specific classes for pruning '''
    mapping = {k: v for k, v in zip(target_class, range(1, len(target_class) + 1))}

    length = int(len(y_train) / (num_cls * 10) - len(target_class))

    x_retrain_train = torch.full([0] + [dim for dim in x_train.shape[1:]], fill_value=0.)
    y_retrain_train = torch.full([0], fill_value=0, dtype=torch.uint8)
    x_retrain_test = torch.full([0] + [dim for dim in x_train.shape[1:]], fill_value=0.)
    y_retrain_test = torch.full([0], fill_value=0, dtype=torch.uint8)


    for i in range(num_cls):
        if i in target_class:
            new_label = mapping[i]

            positive_image = x_train[y_train == i]
            x_retrain_train = torch.concat([x_retrain_train, positive_image], dim=0)
            positive_label = torch.tensor([new_label] * len(y_train[y_train == i]), dtype=torch.uint8)
            y_retrain_train = torch.concat([y_retrain_train, positive_label], dim=0)
            
            positive_image = x_test[y_test == i]
            x_retrain_test = torch.concat([x_retrain_test, positive_image], dim=0)
            positive_label = torch.tensor([new_label] * len(y_test[y_test == i]), dtype=torch.uint8)
            y_retrain_test = torch.concat([y_retrain_test, positive_label], dim=0)

        else:
            negative_image = x_train[y_train == i]
            temp_l = len(negative_image) if len(negative_image) <= length else length
            rnd_idx = torch.randperm(len(y_train[y_train == i]))[:temp_l]
            negative_image = negative_image[rnd_idx]
            negative_label = torch.zeros([temp_l], dtype=torch.uint8)
            x_retrain_train = torch.concat([x_retrain_train, negative_image], dim=0)
            y_retrain_train = torch.concat([y_retrain_train, negative_label], dim=0)

            negative_image = x_test[y_test == i]
            temp_l = len(negative_image) if len(negative_image) <= length else length
            rnd_idx = torch.randperm(len(y_test[y_test == i]))[:temp_l]
            negative_image = negative_image[rnd_idx]
            negative_label = torch.zeros([temp_l], dtype=torch.uint8)
            x_retrain_test = torch.concat ([x_retrain_test, negative_image], dim=0)
            y_retrain_test = torch.concat([y_retrain_test, negative_label], dim=0)
    
    return x_retrain_train, y_retrain_train, x_retrain_test, y_retrain_test

def spike_count(layers, data):
    cnt = 0
    inputs = data
    for layer in layers:
        inputs = layer(inputs)
        if isinstance(layer, (neuron.LIFNode, nn.ReLU)):
            cnt += inputs.count_nonzero()

    return cnt, inputs


def spike_count_per_layer(model, data, T, encoder=None, dataset_type='static'):
    """
    逐层统计 spike 数量并计算 SynOPs。
    返回: {
        'layer_stats': [{name, spikes, fan_out, synops}, ...],
        'total_spikes': int,
        'total_synops': int,
    }
    """
    from spikingjelly.activation_based import functional, layer as sj_layer

    hooks = []
    layer_spike_counts = {}
    layer_fan_outs = {}

    def _make_hook(name):
        def hook_fn(module, inp, out):
            if isinstance(out, torch.Tensor):
                layer_spike_counts.setdefault(name, 0)
                layer_spike_counts[name] += int(out.count_nonzero().item())
        return hook_fn

    conv_idx = 0
    for name, m in model.named_modules():
        if isinstance(m, (neuron.LIFNode, nn.ReLU)):
            tag = f'spike_{conv_idx}'
            hooks.append(m.register_forward_hook(_make_hook(tag)))

            parent_conv = None
            for n2, m2 in model.named_modules():
                if isinstance(m2, (sj_layer.Conv2d, nn.Conv2d)):
                    parent_conv = m2
                elif m2 is m and parent_conv is not None:
                    break
            if parent_conv is not None:
                fan_out = parent_conv.out_channels * parent_conv.kernel_size[0] * parent_conv.kernel_size[1]
            else:
                fan_out = 1
            layer_fan_outs[tag] = fan_out
            conv_idx += 1

    model.eval()
    with torch.no_grad():
        if dataset_type in ['cifar10_dvs', 'nmnist', 'ncaltech']:
            data = data.transpose(0, 1)
            for t in range(T):
                _ = model(data[t])
        else:
            for t in range(T):
                inp = encoder(data) if encoder is not None else data
                _ = model(inp)

    functional.reset_net(model)
    for h in hooks:
        h.remove()

    layer_stats = []
    total_spikes = 0
    total_synops = 0
    for tag in sorted(layer_spike_counts.keys(), key=lambda x: int(x.split('_')[1])):
        spikes = layer_spike_counts[tag]
        fan_out = layer_fan_outs.get(tag, 1)
        synops = spikes * fan_out
        layer_stats.append({
            'name': tag,
            'spikes': spikes,
            'fan_out': fan_out,
            'synops': synops,
        })
        total_spikes += spikes
        total_synops += synops

    return {
        'layer_stats': layer_stats,
        'total_spikes': total_spikes,
        'total_synops': total_synops,
    }

# audio GTZAN
def get_gtzan_dataset(path='./GTZAN/'):
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    num_div=8

    path = f'{path}genres_original'

    music_dataset = []
    genre_target = []
    for root, _, files in os.walk(path):
        for name in files:
            filename = os.path.join(root, name)
            if filename != f'{path}/jazz/jazz.00054.wav':
                music_dataset.append(filename)
                genre_target.append(filename.split("/")[3])

    mel_spec=[]
    genre_new=[]
    for idx, wav in enumerate(music_dataset):
        y, sfr = librosa.load(wav)
        div= np.split(y[:660000], num_div) # different length, preserve the first 660000 features
        for chunck in div:
            melSpec = librosa.feature.melspectrogram(y=chunck, sr=sfr, n_mels=N_MELS,hop_length=HOP_LEN, n_fft=N_FFT)
            melSpec_dB = librosa.power_to_db(melSpec, ref=np.max)
            mel_spec.append(melSpec_dB)
            genre_new.append(genre_target[idx])

    labels = os.listdir(f'{path}')
    # idx2labels = {k: v for k, v in enumerate(labels)}
    labels2idx = {v: k for k, v in enumerate(labels)}   

    genre_id = [labels2idx[item] for item in genre_new]

    X_train, X_test, y_train, y_test = train_test_split(mel_spec, genre_id, test_size=0.2)
    X_train, X_test = np.array(X_train), np.array(X_test)

    # unsqueeze to create channel dimension
    X_train = torch.FloatTensor(X_train).unsqueeze(1)
    X_test = torch.FloatTensor(X_test).unsqueeze(1)

    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    return train_dataset, test_dataset

# audio urbansound
def get_urbansound_dataset(path='./UrbanSound/'):
    up_height = 40
    up_width = 173

    X = []
    y = []

    df = pd.read_csv(os.path.join(path, 'UrbanSound8K.csv'))
    for idx, row in df.iterrows():
        if (idx + 1) % 500 == 0 or (idx + 1) == len(df):
            print(f'Finish {idx + 1} files resample')
        file_name = row['slice_file_name']
        folder_num = row['fold']
        label = row['classID']

        fp = os.path.join(path, f'fold{folder_num}', file_name)
        raw , sr = librosa.load(fp, res_type='kaiser_fast') # pip install resampy
        X_ = librosa.feature.mfcc(y=raw, sr=sr, n_mfcc=40)
        up_points = (up_width, up_height)
        X_ = cv2.resize(X_, up_points, interpolation=cv2.INTER_LINEAR)
        X.append(X_)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    X_train , X_test , y_train , y_test = train_test_split(X, y, test_size=0.2)

    # create channel
    X_train = torch.FloatTensor(X_train).unsqueeze(1)
    X_test = torch.FloatTensor(X_test).unsqueeze(1)

    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    return train_dataset, test_dataset


# get_esc50_dataset 函数将被弃用，因此可以删除或注释掉
# def get_esc50_dataset(path='./ESC-50/'):
#     # 完全模仿 urbansound 的处理流程
#     up_height = 40  # n_mfcc, 与 urbansound 保持一致
#     up_width = 216  # 适配 ESC-50 的5秒音频长度

#     X = []
#     y = []

#     df = pd.read_csv(os.path.join(path, 'esc50.csv'))
#     # 获取用于划分训练/测试集的fold信息
#     folds = df['fold'].to_numpy()
    
#     for idx, row in df.iterrows():
#         if (idx + 1) % 500 == 0 or (idx + 1) == len(df):
#             print(f'Finish {idx + 1} files resample')
        
#         file_name = row['filename']
#         label = row['target']

#         # 修正：音频文件的真实路径包含两层 'audio'
#         fp = os.path.join(path, 'audio', 'audio', file_name)
#         try:
#             raw , sr = librosa.load(fp, res_type='kaiser_fast')
#             # 1. 提取MFCC特征
#             X_ = librosa.feature.mfcc(y=raw, sr=sr, n_mfcc=up_height)
#             # 2. 使用cv2.resize强制统一尺寸
#             up_points = (up_width, up_height)
#             X_ = cv2.resize(X_, up_points, interpolation=cv2.INTER_LINEAR)
#             X.append(X_)
#             y.append(label)
#         except Exception as e:
#             print(f'Error processing file {fp}: {e}')

#     X = np.array(X)
#     y = np.array(y)

#     # 使用ESC-50官方推荐的 fold 5 作为测试集
#     test_mask = (folds == 5)
#     train_mask = ~test_mask

#     X_train, y_train = X[train_mask], y[train_mask]
#     X_test, y_test = X[test_mask], y[test_mask]

#     # create channel
#     X_train = torch.FloatTensor(X_train).unsqueeze(1)
#     X_test = torch.FloatTensor(X_test).unsqueeze(1)

#     y_train = torch.LongTensor(y_train)
#     y_test = torch.LongTensor(y_test)

#     train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
#     test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

#     return train_dataset, test_dataset

    