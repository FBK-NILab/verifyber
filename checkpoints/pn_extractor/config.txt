[DEFAULT]
########### model ###########
model = pn_geom
batch_norm = y
dropout = n
spatial_tn = n
knngraph = 5
k_dec = 5
pool_op = max

########## training ###########
n_epochs = 1000

optimizer = adam
accumulation_interval = n
lr_type = step
learning_rate = 1e-3
min_lr = 5e-5
lr_ep_step = 90
lr_gamma = 0.7
momentum = 0.9
patience = 100

weight_decay = 5e-4
weight_init = n

bn_decay = n
bn_decay_init = 0.5
bn_decay_step = 90
bn_decay_gamma = 0.5

########## loss ###########
loss = nll
nll_w = n

########### general ###########
val_in_train = y
val_freq = 20

n_workers = 6

model_dir = models
save_model = y
save_pred = n

seed = 10

########### logging ###########
verbose = n
print_bwgraph = n

####### ************************************************************************
####### ************************************************************************
####### ************************************************************************

[HCP20]
########### data ###########
dataset = hcp20_graph
#dataset_dir = /home/pietro/datasets/ExTractor_PRIVATE/derivatives/merge_shuffle_trk
dataset_dir = /home/pietro/datasets/ExTractor_PRIVATE/derivatives/streamlines_resampled_16_gt20mm
#dataset_dir = /home/pa/data/ExTractor_PRIVATE/derivatives/streamlines_resampled_16
fixed_size = 8000
#val_dataset_dir = /home/pietro/datasets/ExTractor_PRIVATE/derivatives/merge_shuffle_trk
val_dataset_dir = /home/pietro/datasets/ExTractor_PRIVATE/derivatives/streamlines_resampled_16_gt20mm
#val_dataset_dir = /home/pa/data/ExTractor_PRIVATE/derivatives/streamlines_resampled_16
sub_list_train = data/sub_list_HCP_train.txt
sub_list_val = data/sub_list_HCP_val.txt
sub_list_test = data/sub_list_HCP_test.txt
data_dim = 3
embedding_size = 40
fold_size = 2
return_edges = y

batch_size = 2
repeat_sampling = 3
shuffling = y
rnd_sampling = y
standardization = n
centering = n
n_classes = 2
ignore_class = 0
same_size = y

#experiment_name = gcn2_bn_loss-nll_data-hcp20_coords_fs8000
experiment_name = pngeom2_loss_nll-data_hcp20_gt20mm_resampled16_fs8000_balanced_sampling
#experiment_name = sdec_glob-gat-oriknn_kglob25_loss-nll_data-hcp20_16pts_fs8000

####### ************************************************************************
####### ************************************************************************
####### ************************************************************************
