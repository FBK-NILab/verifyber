import os
import sys
import torch
from models import (DEC, BiLSTM, DECSeq, PN, FINTA)

def get_model(cfg):

    num_classes = int(cfg['n_classes'])
    input_size = int(cfg['data_dim'])

    if cfg['model'] == 'finta':
        classifier = FINTA(input_size,
                            n_classes=num_classes)
    if cfg['model'] == 'blstm':
        classifier = BiLSTM(input_size,
                            n_classes=num_classes,
                            embedding_size=128,
                            hidden_size=256,
                            dropout=False)
    if cfg['model'] == 'sdec':
        classifier = DECSeq(
            input_size,
            num_classes,
            dropout=False,
            k=cfg['k'],
            aggr='max',
            pool_op='max')
    if cfg['model'] == 'dec':
        classifier = DEC(
            input_size,
            num_classes,
            k=cfg['k'],
            aggr='max',
            pool_op='max')
    elif cfg['model'] == 'pn_geom':
        classifier = PN(input_size,
                           num_classes,
                           embedding_size=40)
    return classifier

def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight.data)
        m.bias.data.zero_()

def count_parameters(model):
    print([p.size() for p in model.parameters()])
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_net_graph(classifier, loss, logdir):
    from torchviz import make_dot, make_dot_from_trace
    g = make_dot(loss, params=dict(classifier.named_parameters()))
    g.view('net_bw_graph')

    print('classifier parameters: %d' % int(count_parameters(classifier)))
    os.system('rm -r runs/%s' % logdir.split('/', 1)[1])
    os.system('rm -r tb_logs/%s' % logdir.split('/', 1)[1])
    sys.exit()
