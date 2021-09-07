# MOOD_CGV

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
