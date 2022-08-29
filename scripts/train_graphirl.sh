embodiment="gripper"
algo="graphirl"
dataset="xmagical_diverse"
device="cuda:0"

python src/train_graphirl.py \
--embodiment ${embodiment} \
--algo ${algo} \
--dataset ${dataset}
--device ${device}