import argparse
import math
import os
import shutil
import time
import warnings
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.utils.data import dataset
import apex
from apex.parallel.LARC import LARC
from torchvision.datasets import ImageFolder

from utils.misc import bool_flag, initialize_exp, fix_random_seeds, AverageMeter, restart_from_checkpoint
from backends.pytorch import init_distributed_mode
from datasets import MultiCropDataset, DatasetBaseUnsupervised_HDF5
import models.archs as resnet_models
from models import SwAV, SwAVNet, MoCo, MoCoNet, SimSiam, SimSiamNet

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

logger = getLogger()

parser = argparse.ArgumentParser(description="Implementation of Unsupervised Face Representation Learning")

#########################
#### data parameters ####
#########################
parser.add_argument("--dataset", type=str, choices=["imagenet", "vggface2", "vggface21m", "affectnet", "flickr", "all"])
parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                    help="path to dataset repository")
parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
                    help="list of number of crops (example: [2, 6])")
parser.add_argument("--size_crops", type=int, default=[224], nargs="+",
                    help="crops resolutions (example: [224, 96])")
parser.add_argument("--min_scale_crops", type=float, default=[0.14], nargs="+",
                    help="argument in RandomResizedCrop (example: [0.14, 0.05])")
parser.add_argument("--max_scale_crops", type=float, default=[1], nargs="+",
                    help="argument in RandomResizedCrop (example: [1., 0.14])")
parser.add_argument("--use_pil_blur", type=bool_flag, default=True,
                    help="""use PIL library to perform blur instead of opencv""")   
parser.add_argument("--color_distorsion_scale", type=float, default=1.0)         

#########################
## swav specific params #
#########################
parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                    help="list of crops id used for computing assignments")
parser.add_argument("--temperature", default=0.1, type=float,
                    help="temperature parameter in training loss")
parser.add_argument("--epsilon", default=0.05, type=float,
                    help="regularization parameter for Sinkhorn-Knopp algorithm")
parser.add_argument("--improve_numerical_stability", default=False, type=bool_flag,
                    help="improves numerical stability in Sinkhorn-Knopp algorithm")
parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                    help="number of iterations in Sinkhorn-Knopp algorithm")
parser.add_argument("--feat_dim", default=128, type=int,
                    help="feature dimension")
parser.add_argument("--nmb_prototypes", default=3000, type=int,
                    help="number of prototypes")
parser.add_argument("--queue_length", type=int, default=0,
                    help="length of the queue (0 for no queue)")
parser.add_argument("--epoch_queue_starts", type=int, default=15,
                    help="from this epoch, we start using a queue")

#########################
#### moco parameters  ###
#########################
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--mlp', type=bool_flag, default=False,
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=100, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=64, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--base_lr", default=4.8, type=float, help="base learning rate")
parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
parser.add_argument("--freeze_prototypes_niters", default=313, type=int,
                    help="freeze the prototypes during this many iterations from the start")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
parser.add_argument("--start_warmup", default=0, type=float,
                    help="initial warmup learning rate")

#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url", default="", type=str, help="""url used to set up distributed
                    training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")

#########################
#### other parameters ###
#########################
parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
parser.add_argument("--algorithm", default="swav", type=str, choices=["swav", "moco", "simsiam"])
parser.add_argument("--workers", default=4, type=int,
                    help="number of data loading workers")
parser.add_argument("--checkpoint_freq", type=int, default=25,
                    help="Save the model periodically")
parser.add_argument("--use_fp16", type=bool_flag, default=True,
                    help="whether to train with mixed precision or not")
parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
parser.add_argument("--syncbn_process_group_size", type=int, default=8)
parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")
parser.add_argument("--tensorboard-log-dir", type=str, default="/tensorboard")
parser.add_argument('--print_freq', type=int, default=50 )

def main():
    global args
    args = parser.parse_args()
    init_distributed_mode(args)
    fix_random_seeds(args.seed)
    logger, tensorboard_logger, training_stats = initialize_exp(args, "epoch", "loss")

    # build data
    list_of_datatset = []
    if args.dataset == "imagenet":
        data_path = os.path.join(args.data_path, 'train')
        base_dataset = ImageFolder(data_path)
        list_of_datatset.append(base_dataset)

    if args.dataset == "vggface2" or args.dataset == "vggface21m" or args.dataset == 'all':
        base_dataset = build_dataloder(args, 'vggface2_train.h5')
        list_of_datatset.append(base_dataset)

    if args.dataset == "300wlp" or args.dataset == 'all':
        base_dataset = build_dataloder(args, '300w_lp.h5')
        list_of_datatset.append(base_dataset)

    if args.dataset == "widerface"  or args.dataset == 'all':
        base_dataset = build_dataloder(args, 'widerface.h5')
        list_of_datatset.append(base_dataset)

    if args.dataset == "imdbface" or args.dataset == 'all':
        base_dataset = build_dataloder(args, 'imdbface.h5')
        list_of_datatset.append(base_dataset)

    if args.dataset == "affectnet" or args.dataset == 'all':
        base_dataset = build_dataloder(args, 'affectnet_train.h5')
        list_of_datatset.append(base_dataset)

    if args.dataset == "flickr" or args.dataset == 'all':
        base_dataset = build_dataloder(args, 'flickr_train.h5')
        list_of_datatset.append(base_dataset)


    if args.dataset == 'all':
        assert len(list_of_datatset) > 1
        base_dataset = dataset.ConcatDataset(list_of_datatset)

    train_dataset = MultiCropDataset(
        base_dataset,
        args.size_crops,
        args.nmb_crops,
        args.min_scale_crops,
        args.max_scale_crops,
        pil_blur=args.use_pil_blur,
        color_distorsion_scale=args.color_distorsion_scale
    )
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info("Building data done with {} images loaded.".format(len(train_dataset)))

    # build model
    if args.algorithm == "swav":
        base_model = resnet_models.__dict__[args.arch](num_experts=args.num_experts)
        model = SwAVNet(base_model, 
            projection_hidden_size=2048 if args.arch == "resnet50" else 2048 * int(args.arch[-1]),
            num_prototypes=args.nmb_prototypes, projection_size=args.feat_dim
            )
    elif args.algorithm == "moco":
        base_model = resnet_models.__dict__[args.arch]
        model = MoCoNet(base_model, dim=args.moco_dim, K=args.moco_k, m=args.moco_m, T=args.moco_t, mlp=args.mlp,
            num_experts=args.num_experts)
    elif args.algorithm == "simsiam":
        base_model = resnet_models.__dict__[args.arch](num_experts=args.num_experts, zero_init_residual=True)
        model = SimSiamNet(base_model, dim=args.feat_dim)
        
    # synchronize batch norm layers
    if args.sync_bn == "pytorch":
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif args.sync_bn == "apex":
        process_group = apex.parallel.create_syncbn_process_group(args.syncbn_process_group_size)
        model = apex.parallel.convert_syncbn_model(model, process_group=process_group)
    # copy model to GPU
    model = model.cuda()
    if args.rank == 0:
        logger.info(model)
    logger.info("Building model done.")

    # build optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.wd,
    )
    if args.algorithm == "swav":
        optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
    warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                         math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    logger.info("Building optimizer done.")

    # init mixed precision
    if args.use_fp16:
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")
        logger.info("Initializing mixed precision done.")

    # wrap model
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu_to_work_on]
    )

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
        amp=apex.amp,
    )
    start_epoch = to_restore["epoch"]

    # create the trainer
    if args.algorithm == "swav":
        trainer = SwAV(model, args)
    elif args.algorithm == "moco":
        trainer = MoCo(model, args)
    elif args.algorithm == "simsiam":
        trainer = SimSiam(model, args)

    cudnn.benchmark = True
    queue_path = os.path.join(args.dump_path, "queue" + str(args.rank) + ".pth")

    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # set sampler
        train_loader.sampler.set_epoch(epoch)

        trainer.on_epoch_start(epoch)

        # train the network
        scores, queue = train(train_loader, trainer, optimizer, epoch, lr_schedule, tensorboard_logger)
        training_stats.update(scores)

        # save checkpoints
        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if args.use_fp16:
                save_dict["amp"] = apex.amp.state_dict()
            torch.save(
                save_dict,
                os.path.join(args.dump_path, "checkpoint.pth.tar"),
            )
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                shutil.copyfile(
                    os.path.join(args.dump_path, "checkpoint.pth.tar"),
                    os.path.join(args.dump_checkpoints, "ckp-" + str(epoch) + ".pth"),
                )
        if args.algorithm == "swav" and queue is not None:
            torch.save({"queue": queue}, queue_path)


def train(train_loader, trainer, optimizer, epoch, lr_schedule, tensorboard_logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    trainer.model.train()
    end = time.time()

    for it, inputs in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # update learning rate
        iteration = epoch * len(train_loader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[iteration]

        loss = trainer(inputs)

        # ============ backward and optim step ... ============
        optimizer.zero_grad()
        if args.use_fp16:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # cancel some gradients
        if args.algorithm == "swav" and iteration < args.freeze_prototypes_niters:
            for name, p in trainer.model.named_parameters():
                if "prototypes" in name:
                    p.grad = None
        optimizer.step()

        # ============ misc ... ============
        losses.update(loss.item(), inputs[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank == 0 and it % args.print_freq == 0:
            if args.algorithm == "swav":
                curr_lr = optimizer.optim.param_groups[0]["lr"]
            else:
                curr_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=curr_lr,
                )
            )
            # tensorboard_logger.add_scalar('train_loss', losses)

    return (epoch, losses.avg), trainer.queue

def build_dataloder(args, database_name):
    if not args.data_path.endswith(".h5"):
        data_path = os.path.join(args.data_path, database_name)
    else:
        data_path = args.data_path
    base_dataset = DatasetBaseUnsupervised_HDF5(data_path)
    
    return base_dataset


if __name__ == "__main__":
    main()
