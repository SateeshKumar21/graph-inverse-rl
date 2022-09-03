algo="graphirl"
pretrain_type="cross"
env_type="standard"
run_path="logs/xmagical"
seeds="0,5"
device="cuda:0"

python src/train_policy.py \
--pretrain_algo ${algo} \
--pretrain_type ${pretrain_type} \
--env_type ${env_type} \
--run_path ${run_path} \
--seeds ${seeds} \
--device ${device}