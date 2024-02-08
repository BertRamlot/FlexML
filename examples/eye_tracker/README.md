# EyeTracker

Predict where you are looking at on your screen based on your webcam.

Strong focus on being able to do everything at once, i.e. you can generate data, train the model, and run inference (including GUI) all at the same time.

Features:

- Multiple sources of ground truth available.
- Support for multiple people in frame at once.
- Model can be trained "on the go".
- 

## Results

Loss function is the euclidean distance without clamping to the edges of the screen. Distance is conveyed as a percentage of the width of the screen.
Results vary greatly depending on your setup, a reasonable best and worst case is discussed.

### Setup 1 (best-case)

- FOV: ~53 degrees (distance to screen = screen width)
- No glasses, even lighting
- No head movement (both rotationally and laterally)

Results:

- Test loss: ~X% , corresponds  ()


### Setup 2 (worst-case)

- FOV: ~28 degrees (distance to screen = 2 * screen width)
- Glasses with substantial reflections, uneven ligthing
- Substantial head movement (both rotationally and laterally)

Results:

- Test loss: ~X%

## Usage

Overview arguments:

- `--load_datasets [DATASET1 DATASET2 ...]`
- `--save_dataset [DATASET]`
- `--train`: enables training (off by default)
- `--inference`: enables inference (off by default)
- `--gt_source [SOURCE]`
- `--img_source [SOURCE]`
- `--model [MODEL]`: Model name, 
- `--device [DEVICE]`: torch device

Example usage:

```bash
python3 -m examples.eye_tracker.main --load_dataset my_dataset --save_dataset my_dataset --img_source webcam --gt_source simple-ball --model myModel --train --inference
```

This will:

- 

## Architecture

```mermaid
graph TD;
    subgraph img_thread["Img source Thread"]
        img_src[Webcam Source]
    end
    subgraph gt_thread["Ground truth source Thread"]
        gt_src[Feedback Ball Source]
    end
    sample_muxer[Sample Muxer]
    model_cntrl[Model Controller]
    subgraph model_thread["Model Thread"]
        model_ele[Model]
    end
    load_sample_coll[Load Sample Collection]
    save_sample_coll[Save Sample Collection]
    subgraph face_detector_thread["Face detector Thread"]
        gaze_to_face_convertor[Face Detector]
    end
    overlay[Overlay]
    as_train_sample[To tensor &#40train/val/test&#41]
    as_inference_sample[To tensor &#40inference&#41]
    explor_exploit[Exploitation/Exploratioon\nController]
    disk_input[(Disk)]
    disk_output[(Disk)]
    sample_filter(More than 1 second passed since last save?\nAND\nDoes sample have ground truth &#40i.e. 'y'&#41?)

    img_src-->|image|sample_muxer
    gt_src-->|ball position|sample_muxer
    gt_src-->|ball position|overlay
    sample_muxer-->|gaze sample|gaze_to_face_convertor
    gaze_to_face_convertor-->|face sample|sample_filter
    sample_filter-.->|&#60if yes&#62: face sample|save_sample_coll
    sample_filter-.->|&#60if yes&#62: face sample|as_train_sample
    gaze_to_face_convertor-->|face sample|as_inference_sample
    save_sample_coll-->|csv & images|disk_output
    disk_input-->|csv & images|load_sample_coll
    load_sample_coll-->|face sample|as_train_sample
    as_train_sample-->|&#40X, y, type&#41|model_cntrl
    as_inference_sample-->|&#40X,&#41|model_cntrl
    model_cntrl<-->model_ele
    model_cntrl-->|loss per sample|explor_exploit
    explor_exploit-->gt_src
    model_ele-->|predicted values inference|overlay
```

## Misc

- Epoch numbers are highly inflated as the training data grows over time. Data added later will be used far less overall than the data initially added.
- Support for a form of [Active learning](https://en.wikipedia.org/wiki/Active_learning_(machine_learning)), e.g. move the ball towards areas with high errors or the are undersampled, 