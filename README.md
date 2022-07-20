# PointFix: Learning to Fix Domain Bias for Robust Online Stereo Adaptation(PointFix)

Kwonyoung Kim<sup>1</sup> Jungin Park<sup>1</sup> [Jiyoung Lee](https://lee-jiyoung.github.io/)<sup>2</sup> Dongbo Min<sup>3</sup> Kwanghoon Sohn<sup>1</sup>

<sup>1</sup><sub>[Yonsei University](https://www.yonsei.ac.kr)</sub><br>
<sup>2</sup><sub>[NAVER AI LAB](https://naver-career.gitbook.io/en/teams/clova-cic)</sub><br>
<sup>3</sup><sub>[Ewha University](https://www.ewha.ac.kr)</sub>



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
