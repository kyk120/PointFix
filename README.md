# PointFix: Learning to Fix Domain Bias for Robust Online Stereo Adaptation(PointFix) ECCV 2022

Kwonyoung Kim<sup>1</sup> Jungin Park<sup>1</sup> [Jiyoung Lee](https://lee-jiyoung.github.io/)<sup>2</sup> Dongbo Min<sup>3</sup> Kwanghoon Sohn<sup>1</sup>

<sup>1</sup><sub>[Yonsei University](https://www.yonsei.ac.kr)</sub><br>
<sup>2</sup><sub>[NAVER AI LAB](https://naver-career.gitbook.io/en/teams/clova-cic)</sub><br>
<sup>3</sup><sub>[Ewha University](https://www.ewha.ac.kr)</sub>


This software has been tested with python3 and tensorflow 1.13.1

We employed pretrained initial weights for PointFix training.\\
Pretrained MADNet weights for PointFix training can be found in the [Real-Time Self-Adaptive Deep Stereo](https://github.com/CVLAB-Unibo/Real-time-self-adaptive-deep-stereo) page.

Example of PointFix training of MADNet:
```bash
OUTPUT="path/to/output/folder"
DATASET="./example_dataset.csv"
BATCH_SIZE="4"
ITERATIONS=30000
PRETRAINED_WEIGHTS="./pretrained_nets/MADNet/synthetic/weights.ckpt"
MODEL_NAME="MADNet"
LR="0.0001"
ALPHA="0.00001"
LOSS="mean_l1"
ADAPTATION_LOSS="mean_SSIM_l1"
META_ALGORITHM="PointFix"
RESIZE_SHAPE="380 640"
DATASET_PARAM="Synthia"

python train_pointfix.py --dataset $DATASET -o $OUT_FOLDER -b $BATCH_SIZE -n $ITERATIONS --adaptationSteps $ADAPTATION_ITERATION \
--weights $PRETRAINED_WEIGHTS --lr $LR --alpha $ALPHA --loss $LOSS --adaptationLoss $ADAPTATION_LOSS --unSupervisedMeta \
--metaAlgorithm $META_ALGORITHM --resizeShape $RESIZE_SHAPE --dataset_param $DATASET_PARAM \
--modelName $MODEL_NAME
```


Example of online adaptation test using MADNet and MAD:
```bash
LIST="./example_sequence.csv" 
OUTPUT="path/to/output/folder"
WEIGHTS="path/to/pretrained/network"
MODELNAME="MADNet"
BLOCKCONFIG="block_config/MadNet_full.json"
MODE="MAD"
LR=0.00001

python3 test_online_adaptation.py \
    -l ${LIST} \
    -o ${OUTPUT} \
    --weights ${WEIGHTS} \
    --modelName MADNet \
    --blockConfig ${BLOCKCONFIG} \
    --mode ${MODE} \
    --sampleMode PROBABILITY \
    --logDispStep 1 \
    --lr ${LR}
```
