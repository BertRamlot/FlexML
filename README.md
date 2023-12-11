# EyeTracker

Predicts where you are looking at on your screen based on your webcam.

Features:

- Supports tracking multiple people at once
- Designed to make it easy for anyone to (1) gather data, and (2) train their own model
- Fast data gathering and training, a basic model takes about 2 minutes to create from scratch

## Results

Setup:

- 15.6 inch laptop screen
- ~25 inch viewing distance
- n training samples (generated in x minutes using the moving ball overlay)
- x seconds of training

Results:

- average error of ~6% of the width of my screen (euclidean distance), which is about 2 cm or 0.8 inches

The model is robust against the following phenomena, given that these were adequately represented in the training:

- Uneven lighting
- Glasses (including heavy reflection in those glasses)
- Head rotation, lateral head movement, longitudinal head movement

... but you probably can't use other people's models as your model is overfitted to your screen, webcam, face, ligthing conditions, ...

## Usage

### Creating a dataset

```bash
# Only data gathering
python -m src.overlay --dataset my_dataset
```

### Training a model

```bash
python -m src.train --dataset my_dataset --model my_model
```

Optional arguments ([train.py](./src/train.py)):

- `lr`: learning rate
- `max_epochs`: epoch at which to stop training
- `device`: torch device


### Testing a model
Note: Inference uses the same python file as the data generation. In fact, both can be done at the same time!
```bash
# Only inference
python -m src.overlay --model my_model

# Inference + data gathering
python -m src.overlay --model my_model --dataset my_dataset 
```

Optional inference arguments ([overlay.py](./src/overlay.py)):

- `epoch`: epoch of model, defaults to latest if not passed
- `device`: torch device
