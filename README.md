# EyeTracker

Eye tracking framework for constructing a graph of ML-pipeline-components to:

- generate eye data
- train models
- do demos/inference

Eye tracking specifically refers to "Predicting where you are looking at on your screen based on your webcam."

All of the above can be done simultaniously and in real time, even allowing for feedback from the model to the data generation source (e.g. if a large error exits for samples in a certain area, the model can instruct the data generation tools to sample that area).

Features:

- Supports tracking multiple people at once

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
python overlay.py --dataset my_dataset
```

Controls:

- 'q': Quit the overlay

### Training a model

```bash
python train.py --dataset my_dataset --model my_model
```

Optional arguments ([train.py](./src/train.py)):

- `lr`: learning rate
- `max_epochs`: epoch at which to stop training
- `device`: torch device

### Testing a model

Note: Inference uses the same overlay/python file as the data generation. You can do inference and data gathering at the same time.

```bash
# Only inference
python overlay.py --model my_model

# Inference + data gathering
python overlay.py --model my_model --dataset my_dataset 
```

Optional inference arguments ([overlay.py](./src/overlay.py)):

- `epoch`: epoch of model, defaults to latest if not passed
- `device`: torch device

## TODO

- Make OS independent (i.e. remove "ctypes.windll")µ
