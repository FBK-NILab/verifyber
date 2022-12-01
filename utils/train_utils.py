import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch import optim

from losses.entropy_loss import NLLLoss
from .general_utils import save_dict_to_file

def create_tb_logger(cfg):
    if cfg['experiment_name'] != 'default':
        for ext in range(100):
            exp_name = cfg['experiment_name'] + '_%d' % ext
            logdir = 'runs/%s' % exp_name
            if not os.path.exists(logdir):
                writer = SummaryWriter(log_dir=logdir)
                break
            else:
                writer = SummaryWriter()

    tb_log_name = glob.glob('%s/events*' % logdir)[0].rsplit('/', 1)[1]
    tb_log_dir = 'tb_logs/%s' % exp_name
    os.system('mkdir -p %s' % tb_log_dir)
    os.system('ln -sr %s/%s %s/%s ' %
              (logdir, tb_log_name, tb_log_dir, tb_log_name))

    # os.system('cp configs/main_dsl_config.py %s/config.txt' % (writer.log_dir))
    cfg['cfg_parser_obj'].write(open('%s/config.txt' % (writer.log_dir), 'w'))

    return writer

def get_optimizer(cfg, classifier):
    if cfg['optimizer'] == 'sgd_momentum':
        return optim.SGD(classifier.parameters(),
                         lr=float(cfg['learning_rate']),
                         momentum=float(cfg['momentum']),
                         weight_decay=float(cfg['weight_decay']))
    elif cfg['optimizer'] == 'adam':
        return optim.Adam(classifier.parameters(),
                          lr=float(cfg['learning_rate']),
                          weight_decay=float(cfg['weight_decay']))
    else:
        sys.exit('wrong or unknown optimizer')

def get_lr_scheduler(cfg, optimizer):
    if cfg['lr_type'] == 'step':
        return optim.lr_scheduler.StepLR(optimizer,
                                         int(cfg['lr_ep_step']),
                                         gamma=float(cfg['lr_gamma']))
    elif cfg['lr_type'] == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=float(cfg['lr_gamma']),
            patience=int(cfg['patience']),
            threshold=0.0001,
            min_lr=float(cfg['min_lr']))
    else:
        return None


def get_lr(optimizer):
    for i, param_group in enumerate(optimizer.param_groups):
        return float(param_group['lr'])

def set_lr(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr
        return

def update_bn_decay(cfg, classifier, epoch):
    # inspired by pointnet charlesq34 implementation
    bnd_0 = float(cfg['bn_decay_init'])
    bnd_gamma = float(cfg['bn_decay_gamma'])
    bnd_step = int(cfg['bn_decay_step'])

    bn_momentum = bnd_0 * bnd_gamma**(epoch / bnd_step)
    bn_momentum = 1 - min(0.99, 1 - bn_momentum)
    print('updated bn momentum to %f' % bn_momentum)
    for module in classifier.modules():
        if type(module) == torch.nn.BatchNorm1d:
            module.momentum = bn_momentum

def initialize_loss_dict(cfg):
    loss_dict = {}
    loss_dict[cfg['loss']] = 0.

    return loss_dict

def compute_loss(cfg, logits, target, loss_dict=None):
    tot_loss = 0.
    if cfg['loss'] == 'nll':
        pred = F.log_softmax(logits, dim=-1).view(-1, int(cfg['n_classes']))
        c_w = None if not cfg['nll_w'] else torch.tensor(cfg['nll_w'],
                                                         device=pred.device)
        loss = F.nll_loss(pred, target.long(), weight=c_w)
        tot_loss += loss

    elif cfg['loss'] == 'mse':
        loss = F.mse_loss(logits, target.float(), reduction='mean')
        tot_loss += loss

    elif cfg['loss'] == 'mse_sum':
        loss = F.mse_loss(logits, target.float(), reduction='none')
        loss = loss.sum(-1).mean()
        tot_loss += loss

    elif cfg['loss'] == 'mae':
        loss = F.l1_loss(logits,target.float())
        tot_loss += loss

    if loss_dict is not None:
        loss_dict[cfg['loss']] += loss.item()

    return tot_loss

def log_losses(loss_dict, writer, epoch, n_iters, prefix='train'):
    if loss_dict is None:
        return
    for loss_type, value in loss_dict.items():
        ep_loss = value / n_iters
        writer.add_scalar('%s/%s' % (prefix, loss_type), ep_loss, epoch)

def dump_model(cfg, model, logdir, epoch, score, best=False):
    prefix = ''
    if best:
        prefix = 'best_'

    modeldir = os.path.join(logdir, cfg['model_dir'])
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    else:
        os.system('rm %s/%smodel*.pth' % (modeldir, prefix))
    torch.save(model.state_dict(),
               '%s/%smodel_ep-%d_score-%f.pth' %
                    (modeldir, prefix, epoch, score))

def dump_code(cfg, logdir):
    codedir = os.path.join(logdir, 'train_code/.')
    if not os.path.exists(codedir):
        os.makedirs(codedir)
    source_code_list = [
        'configs', 'datasets', 'loops', 'models', 'losses', 'utils', 'main.py',
        '__init__.py'
    ]
    for source_code in source_code_list:
        os.system('cp -r %s %s' % (source_code, codedir))
