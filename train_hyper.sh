GPU=1
PORT_BASE=6017
GT_PATH=/media/data3/code/lzh/original_data/hypernerf/virg

DATASET=hypernerf
SAVE_PATH=/media/data3/code/lzh/SpatioGS/output

SCENE_LIST=(
    3dprinter
    banana
    broom
    chicken
)
for SCENE in "${SCENE_LIST[@]}"; do
    echo "scene: $SCENE"
    CONFIG=$SCENE
    CUDA_VISIBLE_DEVICES=$GPU python train.py -s $GT_PATH/$SCENE/ --port $(expr $PORT_BASE + $GPU)  --expname $DATASET/$SCENE --configs arguments/$DATASET/$CONFIG.py
    CUDA_VISIBLE_DEVICES=$GPU python render.py --model_path $SAVE_PATH/$DATASET/$CONFIG/  --skip_train --configs arguments/$DATASET/$CONFIG.py
    CUDA_VISIBLE_DEVICES=$GPU python metrics.py --model_path $SAVE_PATH/$DATASET/$CONFIG
done