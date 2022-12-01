import numpy as np
import torch
import torch.nn.functional as F

from utils.data.data_utils import get_dataset, get_gbatch_sample
from utils.general_utils import (
    initialize_metrics,
    log_avg_metrics,
    update_metrics,
    get_metrics_inline,
)
from utils.model_utils import get_model
from utils.train_utils import (
    compute_loss,
    create_tb_logger,
    dump_code,
    dump_model,
    get_lr,
    get_lr_scheduler,
    get_optimizer,
    initialize_loss_dict,
    log_losses,
    update_bn_decay,
)
from utils.data.transforms import RndSampling


def train_ep(cfg, dataloader, model, optimizer, writer, epoch, n_iter):
    """
    run one epoch of training
    """

    # set the model in train mode
    model.train()

    num_classes = int(cfg["n_classes"])
    num_batch = cfg["num_batch"]

    ep_loss = 0.0
    ep_loss_dict = initialize_loss_dict(cfg)
    metrics = initialize_metrics()

    for i_batch, sample_batched in enumerate(dataloader):

        ### reorganize the batch in term of streamlines
        points = get_gbatch_sample(
            sample_batched, int(cfg["fixed_size"]), cfg["same_size"]
        )

        if "unsupervised" in cfg["task"]:
            target = points.x.data
        else:
            target = points["y"]

        points, target = points.to("cuda"), target.to("cuda")

        ### initialize gradients
        if not cfg["accumulation_interval"] or i_batch == 0:
            optimizer.zero_grad()

        ### forward
        logits = model(points)

        ### minimize the loss
        loss = compute_loss(cfg, logits, target, ep_loss_dict)
        ep_loss += loss.item()
        running_ep_loss = ep_loss / (i_batch + 1)

        loss.backward()

        if int(cfg["accumulation_interval"]) % (i_batch + 1) == 0:
            optimizer.step()
            optimizer.zero_grad
        elif not cfg["accumulation_interval"]:
            optimizer.step()

        ### compute performance
        if cfg["task"] == "classification":
            pred = F.log_softmax(logits, dim=-1).view(-1, num_classes)
            pred_choice = pred.data.max(1)[1].int()
            update_metrics(metrics, pred_choice, target, task=cfg["task"])
            print(
                "[%d: %d/%d] train loss: %f %s"
                % (
                    epoch,
                    i_batch,
                    num_batch,
                    loss.item(),
                    get_metrics_inline(metrics, "last"),
                )
            )
        else:
            update_metrics(metrics, logits.float(), target.float(), task=cfg["task"])
            print(
                "[%d: %d/%d] train loss: %f %s"
                % (
                    epoch,
                    i_batch,
                    num_batch,
                    loss.item(),
                    get_metrics_inline(metrics, "last"),
                )
            )
        n_iter += 1

    ep_loss = ep_loss / (i_batch + 1)
    writer.add_scalar("train/epoch_loss", ep_loss, epoch)
    log_losses(ep_loss_dict, writer, epoch, i_batch + 1)
    log_avg_metrics(writer, metrics, "train", epoch)

    return ep_loss, n_iter


def val_ep(cfg, val_dataloader, model, writer, epoch, best_epoch, best_score):
    """
    run the validation phase when called
    """
    best = False
    num_classes = int(cfg["n_classes"])

    # set model in eval mode
    model.eval()

    with torch.no_grad():
        print("\n\n")

        metrics_val = initialize_metrics()
        ep_loss = 0.0

        for i, data in enumerate(val_dataloader):
            points = get_gbatch_sample(data, int(cfg["fixed_size"]), cfg["same_size"])

            if "unsupervised" in cfg["task"]:
                target = points.x.data
            else:
                target = points["y"]

            points, target = points.to("cuda"), target.to("cuda")

            log_str = "VALIDATION [%d: %d/%d] " % (epoch, i, len(val_dataloader))

            logits = model(points)

            loss = compute_loss(cfg, logits, target)
            ep_loss += loss.item()

            ### compute performance
            if cfg["task"] == "classification":
                ref_metrics = "acc"
                pred = F.log_softmax(logits, dim=-1).view(-1, num_classes)
                pred_choice = pred.data.max(1)[1].int()
                update_metrics(metrics_val, pred_choice, target, task=cfg["task"])
                print(
                    "val min / max class pred %d / %d"
                    % (pred_choice.min().item(), pred_choice.max().item())
                )
                print("# class pred ", len(torch.unique(pred_choice)))
                # writer.add_scalar('val/loss', ep_loss / i, epoch)
            else:
                ref_metrics = "mse"
                update_metrics(
                    metrics_val, logits.float(), target.float(), task=cfg["task"]
                )

            log_str += "loss: %.4f " % loss.item()
            log_str += get_metrics_inline(metrics_val, type="last")
            print(log_str)
            # writer.add_scalar('val/loss', ep_loss / i, epoch)

        log_avg_metrics(writer, metrics_val, "val", epoch)
        epoch_score = torch.tensor(metrics_val[ref_metrics]).mean().item()
        print("VALIDATION AVG: %s" % get_metrics_inline(metrics_val, "avg"))
        print("\n\n")

        if ref_metrics == "acc" and epoch_score > best_score:
            best_score = epoch_score
            best_epoch = epoch
            best = True
        elif ref_metrics == "mse" and epoch_score < best_score:
            best_score = epoch_score
            best_epoch = epoch
            best = True

        if cfg["save_model"]:
            dump_model(cfg, model, writer.log_dir, epoch, epoch_score, best=best)

        return best_epoch, best_score


def train(cfg):

    batch_size = int(cfg["batch_size"])
    n_epochs = int(cfg["n_epochs"])
    sample_size = int(cfg["fixed_size"])

    #### DATA LOADING
    trans_train = []
    trans_val = []
    if cfg["rnd_sampling"]:
        trans_train.append(RndSampling(sample_size, maintain_prop=False))
        # prop_vector=[1, 1]))
        trans_val.append(RndSampling(sample_size, maintain_prop=False))

    dataset, dataloader = get_dataset(cfg, trans=trans_train)
    val_dataset, val_dataloader = get_dataset(cfg, trans=trans_val, train=False)
    # summary for tensorboard
    writer = create_tb_logger(cfg)
    dump_code(cfg, writer.log_dir)

    #### BUILD THE MODEL
    model = get_model(cfg)
    print(model)

    #### SET THE TRAINING
    optimizer = get_optimizer(cfg, model)

    lr_scheduler = get_lr_scheduler(cfg, optimizer)

    model.to("cuda")

    num_batch = len(dataset) / batch_size
    print("num of batches per epoch: %d" % num_batch)
    cfg["num_batch"] = num_batch

    n_iter = 0
    if cfg["task"] == "classification":
        best_pred = 0
    else:
        best_pred = np.inf
    best_epoch = 0
    current_lr = float(cfg["learning_rate"])
    initial_nll_w = cfg["nll_w"]
    for epoch in range(n_epochs + 1):

        # update bn decay
        if cfg["bn_decay"] and epoch != 0 and epoch % int(cfg["bn_decay_step"]) == 0:
            update_bn_decay(cfg, model, epoch)

        if cfg["nll_w_decay"] and epoch % int(cfg["nll_w_decay_step"]) == 0:
            cfg["nll_w"][0] = initial_nll_w[0] * cfg["nll_w_decay"] ** epoch

        loss, n_iter = train_ep(
            cfg, dataloader, model, optimizer, writer, epoch, n_iter
        )

        ### validation during training
        if epoch % int(cfg["val_freq"]) == 0 and cfg["val_in_train"]:
            best_epoch, best_pred = val_ep(
                cfg, val_dataloader, model, writer, epoch, best_epoch, best_pred
            )

        # update lr
        if cfg["lr_type"] == "step" and current_lr >= float(cfg["min_lr"]):
            lr_scheduler.step()
        if cfg["lr_type"] == "plateau":
            lr_scheduler.step(loss)

        current_lr = get_lr(optimizer)
        writer.add_scalar("train/lr", current_lr, epoch)

    writer.close()
