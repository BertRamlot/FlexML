# EyeTracker

Predict where you are looking at on your screen based on your webcam. Strong focus on being able to do everything at once, i.e. you can generate data, train the model, and run inference (including GUI) all at the same time.

Features:

- Multiple sources of ground truth available.
- Support for multiple people in frame at once.
- Model can be trained "on the go".
- Supports [Active learning](https://en.wikipedia.org/wiki/Active_learning_(machine_learning)), made possible by the live training nature.

## Results

Setup:

- 15.6 inch laptop screen
- ~25 inch viewing distance
- 0 starting training samples
- x seconds of tracking the ball

Results:

- average euclidean distance error of ~6% of the width of the screen, corresponds to approximately 0.8 inches (2 cm)

## Usage

```bash
python3 -m examples.eye_tracker.main --load_dataset my_dataset --save_dataset my_dataset --img_source webcam --gt_source simple-ball --model myModel
```

```mermaid
graph TD;
    img_src[Webcam Source]
    gt_src[Feedback Ball Source]
    sample_muxer[Sample Muxer]
    model_cntrl[Model Controller]
    model_ele[Model]
    load_sample_coll[Load Sample Collection]
    save_sample_coll[Save Sample Collection]
    gaze_to_face_convertor[Face Detector]
    overlay[Overlay]
    to_train_sample[To Tensor]
    explor_exploit[Exploitation/Exploratioon\nController]
    disk_input[(Disk)]
    disk_output[(Disk)]

    img_src-->|image|sample_muxer
    gt_src-->sample_muxer
    gt_src-->overlay
    sample_muxer-->gaze_to_face_convertor
    gaze_to_face_convertor-->to_train_sample
    sample_muxer-->save_sample_coll
    save_sample_coll-->disk_output
    disk_input-->load_sample_coll
    load_sample_coll-->gaze_to_face_convertor
    to_train_sample-->model_cntrl
    model_cntrl-->model_ele
    model_cntrl-->|loss per sample|explor_exploit
    explor_exploit-->gt_src
    model_ele-->model_cntrl
    model_cntrl-->|inference & val predictions|overlay
```
