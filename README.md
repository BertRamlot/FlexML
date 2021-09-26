# EyeTracker
Tracks your eyes and predicts where you are looking on the screen.
Comes with pretrained model and tools to generate your own data to finetune the pretrained network to your face.

Trained using pytorch.

## Data generation

Generate your own data face data.

file_name | Description | Noise | Generation speed
--------- | ----------- | ----- | ----------------
ball_tracking_data_generator.py | Records a data points periodically while moving a ball across the screen. We assume you are looking at the tip of the mouse. | Pretty noisy | Very fast
click_listener_data_generator.py | Records a data point when you click your mouse, we assume you are looking at the tip of the mouse. | Almost no noise | Slow


## Training/Running
file_name | Description
--------- | -
train.py  | Trains your model.
demo.py   | Runs your trained model, draws prediction onscreen of where the model thinks you are looking.


## Data visualisation/cleaning + Problem solving

Some extra visualization tools and data manipulation tools are included.

file_name | Description
--------- | -----------
clean_meta_data.py | Removes rows from meta data csv if the corresponding face is not found. Allows you to delete pictures without having to find in csv.
data_visualisation.ipynb | Collection of visulations of your data. Useful to detect faulty data, biggest losses, data distribution, prediction distribution, ...
face_feature_live.py | Displays the output of "shape_predictor_68_face_landmarks.dat", accurate predictions of this model are crucial for training/running.