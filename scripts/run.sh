EXPERIMENT_PATH="output/unsup_pretrain"
mkdir -p $EXPERIMENT_PATH
DATASET_PATH=~/Datasets/imagenet/
DATASET=imagenet

python -m torch.distributed.launch \
--nproc_per_node=8 \
--nnodes=1 \
--node_rank=0 \
--master_addr="127.0.0.1" \
--master_port=40000 \
flr/main.py \
--data_path $DATASET_PATH \
--nmb_crops 2 6 \
--size_crops 224 96 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--use_pil_blur false \
--crops_for_assign 0 1 \
--temperature 0.1 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--feat_dim 128 \
--nmb_prototypes 3000 \
--queue_length 0 \
--epoch_queue_starts 15 \
--epochs 400 \
--batch_size 128 \
--base_lr 4.8 \
--final_lr 0.0048 \
--freeze_prototypes_niters 313 \
--wd 0.000001 \
--warmup_epochs 10 \
--start_warmup 0.3 \
--dist_url "tcp://127.0.0.1:40000" \
--arch resnet50 \
--use_fp16 true \
--sync_bn apex \
--dataset $DATASET \
--dump_path $EXPERIMENT_PATH
