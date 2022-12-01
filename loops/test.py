import glob
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch as gBatch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import datasets as ds
from utils.data.data_utils import (get_dataset, get_gbatch_sample,
                                   get_transforms)
from utils.general_utils import (initialize_metrics, log_avg_metrics,
                                 update_metrics)
from utils.model_utils import get_model, print_net_graph
from utils.train_utils import (compute_loss, create_tb_logger, dump_code,
                               dump_model, get_lr, get_lr_scheduler,
                               get_optimizer, initialize_loss_dict, log_losses,
                               update_bn_decay, set_lr)
from utils.data.transforms import RndSampling, TestSampling, SampleStandardization

def test(cfg):
    num_classes = int(cfg['n_classes'])
    sample_size = int(cfg['fixed_size'])
    cfg['loss'] = cfg['loss'].split(' ')
    batch_size = 1
    cfg['batch_size'] = batch_size
    epoch = eval(str(cfg['n_epochs']))
    #n_gf = int(cfg['num_gf'])
    input_size = int(cfg['data_dim'])

    trans_val = []
    if cfg['rnd_sampling']:
        trans_val.append(TestSampling(sample_size))

    if 'bids' in cfg['dataset']:
        dataset = ds.BIDSDataset(cfg['sub_list_test'],
                                  cfg['dataset_dir'],
                                  run='test',
                                  data_name=cfg['data_name'],
                                  transform=transforms.Compose(trans_val),
                                  return_edges=cfg['return_edges'],
                                  labels_dir=cfg['labels_dir'],
                                  labels_name=cfg['labels_name'],
                                  with_gt=cfg['with_gt'],
                                  split_obj=True,)
    elif 'hcp20' in cfg['dataset']:
        dataset = ds.HCP20Dataset(cfg['sub_list_test'],
                                  cfg['dataset_dir'],
                                  transform=transforms.Compose(trans_val),
                                  with_gt=cfg['with_gt'],
                                  #distance=T.Distance(norm=True,cat=False),
                                  return_edges=True,
                                  split_obj=True,
                                  train=False,
                                  load_one_full_subj=False,
                                  labels_dir=cfg['labels_dir'])
    elif cfg['dataset'] == 'atlasparts':
        dataset = ds.StreamAtlasDataset(cfg['sub_list_test'],
                                        cfg['dataset_dir'],
                                        data_name=cfg['data_name'],
                                        run='test',
                                        labels_dir=cfg['labels_dir'],
                                        lbls_name=cfg['labels_name'],
                                        same_size=cfg['same_size'],
                                        transform=transforms.Compose(trans_val),
                                        with_gt=cfg['with_gt'],
                                        return_edges=cfg['return_edges'],
                                        self_loops=False,
                                        data_per_point=False,
                                        add_tangent=False,
                                        split_obj=True
                                        )
    else:
        sys.exit('Unexisting dataset chosen')


    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)
    print("Validation dataset loaded, found %d samples" % (len(dataset)))

    for ext in range(100):
        logdir = '%s/test_%d' % (cfg['exp_path'], ext)
        if not os.path.exists(logdir):
            break
    writer = SummaryWriter(log_dir=logdir)
    if cfg['weights_path'] == '':
        cfg['weights_path'] = glob.glob(cfg['exp_path'] + '/models/best*')[0]
        epoch = int(cfg['weights_path'].rsplit('-',1)[1].split('.')[0])
    elif 'ep-' in cfg['weights_path']:
        epoch = int(cfg['weights_path'].rsplit('-',1)[1].split('.')[0])

    tb_log_name = glob.glob('%s/events*' % writer.log_dir)[0].rsplit('/',1)[1]
    tb_log_dir = 'tb_logs/%s' % logdir.split('/',1)[1]
    os.system('mkdir -p %s' % tb_log_dir)
    os.system('ln -sr %s/%s %s/%s ' %
                        (writer.log_dir, tb_log_name, tb_log_dir, tb_log_name))

    #### BUILD THE MODEL
    classifier = get_model(cfg)

    classifier.cuda()
    classifier.load_state_dict(torch.load(cfg['weights_path']))
    classifier.eval()

    with torch.no_grad():
        pred_buffer = {}
        sm_buffer = {}
        sm2_buffer = {}
        gf_buffer = {}
        emb_buffer = {}
        print('\n\n')
        if cfg['task'] == 'classification':
            mean_val_acc = torch.tensor([])
            mean_val_iou = torch.tensor([])
            mean_val_prec = torch.tensor([])
            mean_val_recall = torch.tensor([])
        elif cfg['task'] == 'regression':
            mean_val_mse = torch.tensor([])
            mean_val_mae = torch.tensor([])
            mean_val_rho = torch.tensor([])

        if 'split_obj' in dir(dataset) and dataset.split_obj:
            split_obj = True
        else:
            split_obj = False
            dataset.transform = []

        if split_obj:
            consumed = False
        else:
            consumed = True
        j = 0
        visualized = 0
        new_obj_read = True

        while j < len(dataset):
            data = dataset[j]

            if split_obj:
                if new_obj_read:
                    if cfg['task'] == 'classification':
                        obj_pred_choice = torch.zeros(data['obj_full_size'], dtype=torch.int).cuda()
                        obj_target = torch.zeros(data['obj_full_size'], dtype=torch.int).cuda()
                        new_obj_read = False
                    elif cfg['task'] == 'regression':
                        obj_pred_choice = torch.zeros(data['obj_full_size'], dtype=torch.float32).cuda()
                        obj_target = torch.zeros(data['obj_full_size'], dtype=torch.float32).cuda()
                        new_obj_read = False

                if len(dataset.remaining[j]) == 0:
                    consumed = True

            sample_name = data['name'] if type(data['name']) == str else data['name'][0]
            points = gBatch().from_data_list([data['points']])
            #points = data['points']
            if 'bvec' in points.keys:
                points.batch = points.bvec.clone()
                del points.bvec            
            if cfg['with_gt']:
                target = points['y']
                target = target.to('cuda')
            if cfg['same_size']:
                points['lengths'] = points['lengths'][0].item()
            #if cfg['model'] == 'pointnet_cls':
                #points = points.view(len(data['obj_idxs']), -1, input_size)
            points = points.to('cuda')

            if cfg['task'] == 'classification':
                logits = classifier(points)
                logits = logits.view(-1, num_classes)
                pred = F.log_softmax(logits, dim=-1).view(-1, num_classes)
                pred_choice = pred.data.max(1)[1].int()
            elif cfg['task'] == 'regression':
                pred = classifier(points)
            
            if split_obj:
                if cfg['task'] == 'classification':
                    obj_pred_choice[data['obj_idxs']] = pred_choice
                    obj_target[data['obj_idxs']] = target.int()
                elif cfg['task'] == 'regression':
                    obj_pred_choice[data['obj_idxs']] = pred.view(-1)
                    obj_target[data['obj_idxs']] = target.float()
                #if cfg['save_embedding']:
                #    obj_embedding[data['obj_idxs']] = classifier.embedding.squeeze()
            else:
                obj_data = points
                obj_pred_choice = pred_choice
                obj_target = target
                if cfg['save_embedding']:
                    obj_embedding = classifier.embedding.squeeze()

            if cfg['with_gt'] and consumed:
                if cfg['task'] == 'classification':
                    print('val max class pred ', obj_pred_choice.max().item())
                    print('val min class pred ', obj_pred_choice.min().item())
                    obj_pred_choice = obj_pred_choice.view(-1,1)
                    obj_target = obj_target.view(-1,1)
                    correct = obj_pred_choice.eq(obj_target.data.int()).cpu().sum()
                    acc = correct.item()/float(obj_target.size(0))
                    tp = torch.mul(obj_pred_choice.data, obj_target.data.int()).cpu().sum().item()+0.00001
                    fp = obj_pred_choice.gt(obj_target.data.int()).cpu().sum().item()
                    fn = obj_pred_choice.lt(obj_target.data.int()).cpu().sum().item()
                    tn = correct.item() - tp
                    iou = torch.tensor([float(tp)/(tp+fp+fn)])
                    prec = torch.tensor([float(tp)/(tp+fp)])
                    recall = torch.tensor([float(tp)/(tp+fn)])
                    mean_val_prec = torch.cat((mean_val_prec, prec), 0)
                    mean_val_recall = torch.cat((mean_val_recall, recall), 0)
                    mean_val_iou = torch.cat((mean_val_iou, iou), 0)
                    mean_val_acc = torch.cat((mean_val_acc, torch.tensor([acc])), 0)
                    print('VALIDATION [%d: %d/%d] val acc: %f' % (epoch, j, len(dataset), acc))
                    
                elif cfg['task'] == 'regression':
                    print('val max class target ', obj_target.max().item())
                    print('val min class target ', obj_target.min().item())
                    mae = torch.mean(abs(obj_target.data.cpu() - obj_pred_choice.data.cpu())).item()
                    mse = torch.mean((obj_target.data.cpu() - obj_pred_choice.data.cpu())**2).item()
                    rho, pval = spearmanr(obj_target.data.cpu().numpy(),obj_pred_choice.data.cpu().numpy())
                    #np.save(writer.log_dir + '/predictions_'+sample_name+'.npy',obj_pred_choice.data.cpu().numpy()) 
                    mean_val_mae = torch.cat((mean_val_mae, torch.tensor([mae])), 0)
                    mean_val_mse = torch.cat((mean_val_mse, torch.tensor([mse])), 0)
                    mean_val_rho = torch.cat((mean_val_rho, torch.tensor([rho])), 0)
                    print('VALIDATION [%d: %d/%d] val mse: %f val mae: %f val rho: %f' % (epoch, j, len(dataset), mse, mae, rho))
                
            if cfg['save_pred'] and consumed:
                print('buffering prediction %s' % sample_name)
                #sl_idx = np.where(obj_pred.data.cpu().view(-1).numpy() == 1)[0]
                #pred_buffer[sample_name] = sl_idx.tolist()

            if consumed:
                print(j)
                j += 1
                if split_obj:
                    consumed = False
                    new_obj_read = True

    #if cfg['save_pred']:
        #os.system('rm -r %s/predictions_test*' % writer.log_dir)
     #   pred_dir = writer.log_dir + '/predictions_test_%d' % epoch
      #  if not os.path.exists(pred_dir):
       #     os.makedirs(pred_dir)
        #print('saving files')
        #for filename, value in pred_buffer.items():
        #    with open(os.path.join(pred_dir, filename) + '.pkl', 'wb') as f:
        #        pickle.dump(
        #            value, f, protocol=pickle.HIGHEST_PROTOCOL)

    if cfg['with_gt']:
        if cfg['task'] == 'regression':
            print('TEST MSE: %f' % torch.mean(mean_val_mse).item())
            print('TEST MAE: %f' % torch.mean(mean_val_mae).item())
            print('TEST RHO: %f' % torch.mean(mean_val_rho).item())
            final_scores_file = writer.log_dir + '/final_scores_test_%d.txt' % epoch
            scores_file = writer.log_dir + '/scores_test_%d.txt' % epoch
            print('saving scores')
            with open(scores_file, 'w') as f: 
                f.write('mse\n')
                f.writelines('%f\n' % v for v in mean_val_mse.tolist())
                f.write('mae\n')
                f.writelines('%f\n' % v for v in mean_val_mae.tolist())
                f.write('rho\n')
                f.writelines('%f\n' % v for v in mean_val_rho.tolist())
            with open(final_scores_file, 'w') as f:
                f.write('mse\n')
                f.write('%f\n' % mean_val_mse.mean())
                f.write('%f\n' % mean_val_mse.std())
                f.write('mae\n')
                f.write('%f\n' % mean_val_mae.mean())
                f.write('%f\n' % mean_val_mae.std())
                f.write('rho\n')
                f.write('%f\n' % mean_val_rho.mean())
                f.write('%f\n' % mean_val_rho.std())

        elif cfg['task'] == 'classification':
            print('TEST ACCURACY: %f' % torch.mean(mean_val_acc).item())
            print('TEST PRECISION: %f' % torch.mean(mean_val_prec).item())
            print('TEST RECALL: %f' % torch.mean(mean_val_recall).item())
            print('TEST IOU: %f' % torch.mean(mean_val_iou).item())
            mean_val_dsc = mean_val_prec * mean_val_recall * 2 / (mean_val_prec + mean_val_recall)
            final_scores_file = writer.log_dir + '/final_scores_test_%d.txt' % epoch
            scores_file = writer.log_dir + '/scores_test_%d.txt' % epoch
            print('saving scores')
            with open(scores_file, 'w') as f: 
                f.write('acc\n')
                f.writelines('%f\n' % v for v in  mean_val_acc.tolist())
                f.write('prec\n')
                f.writelines('%f\n' % v for v in  mean_val_prec.tolist())
                f.write('recall\n')
                f.writelines('%f\n' % v for v in  mean_val_recall.tolist())
                f.write('dsc\n')
                f.writelines('%f\n' % v for v in  mean_val_dsc.tolist())
            with open(final_scores_file, 'w') as f:
                f.write('acc\n')
                f.write('%f\n' % mean_val_acc.mean())
                f.write('%f\n' % mean_val_acc.std())
                f.write('prec\n')
                f.write('%f\n' % mean_val_prec.mean())
                f.write('%f\n' % mean_val_prec.std())
                f.write('recall\n')
                f.write('%f\n' % mean_val_recall.mean())
                f.write('%f\n' % mean_val_recall.std())
                f.write('dsc\n')
                f.write('%f\n' % mean_val_dsc.mean())
                f.write('%f\n' % mean_val_dsc.std())

    print('\n\n')
