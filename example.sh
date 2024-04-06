
# Prepare the datasets and the checkpoints of the pre-trained neural link predictor
sh script_prepare.sh


# Default settings on each dataset
python3 train_gnn.py \
  --task_folder data/FB15k-237-betae \
  --output_dir log/FB15k-237/clmpt_2_8192_512b_5e-5_mean \
  --checkpoint_path pretrain/cqd/FB15k-237-model-rank-1000-epoch-100-1602508358.pt \
  --agg_func mean \
  --reasoner clmpt \
  --embedding_dim 1000 \
  --hidden_dim 8192 \
  --device cuda:0 \
  --batch_size 512 \
  --learning_rate 5e-5 \
  --num_layers 2

python3 train_gnn.py \
  --task_folder data/NELL-betae \
  --output_dir log/NELL/clmpt_2_8192_512b_5e-5_0.1 \
  --checkpoint_path pretrain/cqd/NELL-model-rank-1000-epoch-100-1602499096.pt \
  --agg_func mean \
  --reasoner clmpt \
  --embedding_dim 1000 \
  --device cuda:1 \
  --hidden_dim 8192 \
  --temp 0.1 \
  --batch_size_eval_dataloader 8 \
  --batch_size_eval_truth_value 1 \
  --batch_size 512 \
  --learning_rate 5e-5 \
  --num_layers 2 \
  --pre_norm

python3 train_gnn.py \
  --task_folder data/FB15k-betae \
  --output_dir log/FB15k/clmpt_2_8192_1024b_1e-4 \
  --checkpoint_path pretrain/cqd/FB15k-model-rank-1000-epoch-100-1602520745.pt \
  --agg_func mean \
  --reasoner clmpt \
  --embedding_dim 1000 \
  --device cuda:1 \
  --hidden_dim 8192 \
  --batch_size 1024 \
  --learning_rate 1e-4 \
  --num_layers 2 \
  --pre_norm


# When considering not using conditional message passing, just change the reasoner, namely --reasoner lmpt
python3 train_gnn.py \
  --task_folder data/FB15k-237-betae \
  --output_dir log/FB15k-237/lmpt_2_8192_512b_5e-5_mean \
  --checkpoint_path pretrain/cqd/FB15k-237-model-rank-1000-epoch-100-1602508358.pt \
  --agg_func mean \
  --reasoner lmpt \
  --embedding_dim 1000 \
  --hidden_dim 8192 \
  --device cuda:0 \
  --batch_size 512 \
  --learning_rate 5e-5 \
  --num_layers 2


# When considering freezing the pre-trained neural link predictor, just add --freeze_nlp
python3 train_gnn.py \
  --task_folder data/FB15k-237-betae \
  --output_dir log/FB15k-237/clmpt_2_8192_512b_5e-5_mean_freeze \
  --checkpoint_path pretrain/cqd/FB15k-237-model-rank-1000-epoch-100-1602508358.pt \
  --agg_func mean \
  --reasoner clmpt \
  --embedding_dim 1000 \
  --hidden_dim 8192 \
  --device cuda:0 \
  --batch_size 512 \
  --learning_rate 5e-5 \
  --num_layers 2 \
  --freeze_nlp


# Summarize the results from log files
python3 read_eval_from_log.py --log_file log/FB15k-237/clmpt_2_8192_512b_5e-5_mean/output.log

