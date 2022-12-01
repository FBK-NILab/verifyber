import os
import torch
from datetime import date
import numpy as np
import random
import configparser

def is_float(val):
    try:
        num = float(val)
    except ValueError:
        return False
    return True

def is_int(val):
    try:
        num = int(val)
    except ValueError:
        return False
    return True

def get_cfg_value(value):
    if value[0] == '[' and value[-1] == ']':
        value = [get_cfg_value(v) for v in value[1:-1].split()]
        return value
    if value == 'y':
        return True
    if value == 'n':
        return False
    if is_int(value):
        return int(value)
    if is_float(value):
        return float(value)
    return value

def set_exp_name(cfg, modelname, dataname):
    exp = cfg['experiment_name']
    exp = exp.replace('DATE', str(date.today()))
    exp = exp.replace('MODEL', modelname.lower())
    exp += '_data-{}'.format(dataname.lower())
    cfg['experiment_name'] = exp
    return

def print_cfg(cfg, fileobj=None):
    for k in sorted(cfg.keys()):
        line = '%s : %s' % (k, cfg[k])
        if fileobj is None:
            print(line)
        else:
            fileobj.write(line + '\n')

def save_dict_to_file(dic, filename):
    f = open(filename, 'w')
    f.write(str(dic))
    f.close()

def load_dict_from_file(filename):
    f = open(filename, 'r')
    data = f.read()
    f.close()
    return eval(data)

def initialize_metrics():
    metrics = {}
    metrics['acc'] = []
    metrics['iou'] = []
    metrics['prec'] = []
    metrics['recall'] = []
    metrics['mse'] = []
    metrics['abse'] = []

    return metrics


def update_metrics(metrics, prediction, target, task='classification'):

    if task == 'classification':
        prediction = prediction.data.int().cpu()
        target = target.data.int().cpu()

        correct = prediction.eq(target).sum().item()
        acc = correct / float(target.size(0))

        tp = torch.mul(prediction, target).sum().item() + 0.00001
        fp = prediction.gt(target).sum().item()
        fn = prediction.lt(target).sum().item()
        tn = correct - tp

        iou = float(tp) / (tp + fp + fn)
        prec = float(tp) / (tp + fp)
        recall = float(tp) / (tp + fn)
        
        metrics['prec'].append(prec)
        metrics['recall'].append(recall)
        metrics['acc'].append(acc)
        metrics['iou'].append(iou)
    else:
        prediction = prediction.data.cpu()
        target = target.data.cpu()

        abs_err = torch.mean(abs(prediction-target))
        mserr = torch.mean((target-prediction)**2)
        
        metrics['abse'].append(abs_err)
        metrics['mse'].append(mserr)


def get_metrics_inline(metrics, type='avg'):
    s = ''
    if type == 'avg':
        s = ', '.join(['%s : %.4f' % (k[:3], torch.tensor(v).mean())
                     for k, v in metrics.items() if len(v) > 0])
    elif type == 'last':
        s = ', '.join(['%s : %.4f' % (k[:3], v[-1])
                     for k, v in metrics.items() if len(v) > 0])
    return s


def log_avg_metrics(writer, metrics, prefix, epoch):
    for k, v in metrics.items():
        if type(v) == list:
            v = torch.tensor(v)
        if len(v) == 0:
            continue
        writer.add_scalar('%s/epoch_%s' % (prefix, k), v.mean().item(), epoch)
        #writer.add_scalar('%s/epoch_%s' % (prefix, k), v.float().mean().item(), epoch)

def batched_cdist_l2(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_nrom = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.baddbmm(x2_norm.transpose(-2, -1),
                        x1,
                        x2.transpose(-2, -1),
                        alpha=-2).add_(x1_norm).clamp_min_(1e-30).sqrt_()
    return res

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    np.random.RandomState(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)