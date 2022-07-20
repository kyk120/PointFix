# PointFix


Example of online adaptation test using MADNet and MAD:
```bash
LIST="path/to/the/list/of/frames/" 
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
    --logDispStep 1
    --lr ${LR}
```
