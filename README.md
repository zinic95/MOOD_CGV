# MOOD_CGV
Official Pytorch implementation code of Self-supervised 3D Out-of-Distribution Detection via Pseudoanomaly Generation [(link)](https://link.springer.com/chapter/10.1007/978-3-030-97281-3_15). 
## Requirements
Install python requirements:

```
pip install -r requirements.txt
```

We suggest the following foloder structure for training
```
data/
--- brain/
------ train/
------ valid/
--- abdom/
------ train/
------ valid/
```

## Run the training
```
For brain
  python train.py --dataset your_brain_folder --category brain
For abdom
  python train.py --dataset your_abdom_folder --category abdom
```

## Build the Docker
Build the docker
```
./build.sh
```

## Citation
You are encouraged to modify/distribute this code. However, please acknowledge this code and cite the paper appropriately.
```
@InProceedings{10.1007/978-3-030-97281-3_15,
author="Cho, Jihoon
and Kang, Inha
and Park, Jinah",
editor="Aubreville, Marc
and Zimmerer, David
and Heinrich, Mattias",
title="Self-supervised 3D Out-of-Distribution Detection via Pseudoanomaly Generation",
booktitle="Biomedical Image Registration, Domain Generalisation and Out-of-Distribution Analysis",
year="2022",
publisher="Springer International Publishing",
address="Cham",
pages="95--103",
isbn="978-3-030-97281-3"
}
```
