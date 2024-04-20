import os
import re
import glob
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed=1024):
    if seed < 0:
        seed = random.randint(0, 100000)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return seed


def cosine_similarity(x, y):
    # x: N x D
    # y: M x D
    cos = nn.CosineSimilarity(dim=0)  # 创建一个计算余弦相似度的对象
    cos_sim = []  # 用于存储所有的余弦相似度矩阵
    for xi in x:  # 对x中的每个向量进行循环
        cos_sim_i = []  # 用于存储一个向量与y中所有向量的余弦相似度
        for yj in y:  # 对y中的每个向量进行循环
            cos_sim_i.append(cos(xi, yj))  # 计算xi和yj之间的余弦相似度，并添加到cos_sim_i中
        cos_sim_i = torch.stack(cos_sim_i)  # 将cos_sim_i中的所有余弦相似度堆叠成一个张量
        cos_sim.append(cos_sim_i)  # 将cos_sim_i添加到cos_sim中
    cos_sim = torch.stack(cos_sim)  # 将cos_sim中的所有余弦相似度矩阵堆叠成一个张量
    return cos_sim  # 返回余弦相似度矩阵，形状为(N, M)


def euclidean_dist_similarity(x, y):
    """x和y分别是大小为N x D和M x D的张量，表示N和M个向量，每个向量有D个元素。函数的输出是一个大小为N x M的张量，
    表示x中的每个向量与y中的每个向量之间的欧几里得距离的负值。代码的实现方式是将x和y扩展为大小为N x M x D的张量，
    然后计算每个向量之间的平方误差和，最后返回结果。"""
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return -torch.pow(x - y, 2).sum(2)  # N*M

def manhattan_distance(x, y):
    # x: N x D
    # y: M x D
    man_dist = []  # 用于存储所有的曼哈顿距离矩阵
    for xi in x:  # 对x中的每个向量进行循环
        man_dist_i = []  # 用于存储一个向量与y中所有向量的曼哈顿距离
        for yj in y:  # 对y中的每个向量进行循环
            man_dist_i.append(torch.sum(torch.abs(xi - yj)))  # 计算xi和yj之间的曼哈顿距离，并添加到man_dist_i中
        man_dist_i = torch.stack(man_dist_i)  # 将man_dist_i中的所有曼哈顿距离堆叠成一个张量
        man_dist.append(man_dist_i)  # 将man_dist_i添加到man_dist中
    man_dist = torch.stack(man_dist)  # 将man_dist中的所有曼哈顿距离矩阵堆叠成一个张量
    return man_dist

def metrics(pred, true):
    pred = np.array(pred).reshape(-1)
    true = np.array(true).reshape(-1)
    # acc
    acc = np.mean((pred == true))
    # f_score
    f_score = f1_score(true, pred, average='macro')
    recall = recall_score(true, pred, average='macro')
    return acc, f_score, recall


def multiclass_mcc(true_labels, predicted_labels):
    """计算每个类别的马修相关系数
    true_labels (numpy.ndarray): 真实标签数组，每个元素是0到4之间的标签
    predicted_labels (numpy.ndarray): 预测标签数组，每个元素是0到4之间的标签"""
    # num_classes = len(np.unique(true_labels))
    # mcc_per_class = np.zeros(num_classes)
    #
    # for i in range(num_classes):
    #     true_mask_i = true_labels == i
    #     predicted_mask_i = predicted_labels == i
    #     mcc_per_class[i] = matthews_corrcoef(true_mask_i, predicted_mask_i)
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    num_classes = len(np.unique(true_labels))
    mcc_per_class = np.zeros(num_classes)

    for i in range(num_classes):
        true_mask_i = true_labels == i
        predicted_mask_i = predicted_labels == i
        mcc_per_class[i] = matthews_corrcoef(true_mask_i.astype(int), predicted_mask_i.astype(int))

    return mcc_per_class


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path