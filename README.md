# viewpoint_estimation


## To Do

<!-- - [x] Convert all links to relative paths!
- [x] Include corrected links to weights! (check!)
- [x] Get RenderForCNN and ClickHere Running (without regards to accuracy)
- [x] Fix Loss Function to match what was in the r4cnn paper (ignore what their code actually did for now -- avoid normalization)
- [x] Test training code with updated loss function -->
<!-- - [ ] Get RenderForCNN weights in .npy (and remove lua version from all code) -- cleanup -->
<!-- - [x] Test suspected modification to map generation (flipping X axis) -> did not work! -->
<!-- - [ ] Update README to reflect current content -->
<!-- - [ ] Normalize Class Naming methodology (Capital vs normal .. ) -->
<!-- - [ ] Try to resolve dependence on util .. be more systematic about my imports! -->
- [ ] Train models from scratch!
- [ ] Test model save and resume
- [ ] Test parallelization
- [ ] Dataset wrapper for Synthetic dataset

## Introduction

This is a [PyTorch](http://pytorch.org) implementation of [Clickhere CNN](https://github.com/rszeto/click-here-cnn)
and [Render for CNN](https://github.com/shapenet/RenderForCNN).

We currently provide the model, converted weights, dataset classes, and training/evaluation scripts.
This implementation also includes an implementation of the Geometric Structure Aware loss function first mentioned in Render For CNN.


If you have any questions, please email me at mbanani@umich.edu.


## Getting Started

### Generating the data
Download the [Pascal 3D+ dataset](http://cvgl.stanford.edu/projects/pascal3d.html) (Release 1.1).
Set the path for the Pascal3D directory in util/Paths.py. Finally, run the following command from the repository's root directory.

        cd data/
        python generate_pascal3d_csv.py

Please note that this will generate the csv files for 4 datasets: Pascal 3D+ (full), Pascal 3D+ (easy), and Pascal 3D-Vehicles (with keypoints). Those datasets are needed to obtain different sets of results.

### Pre-trained Model Weights

We have converted the RenderForCNN and Clickhere model weights from the respective caffe models.
The converted models can be downloaded here
([Render For CNN weights](http://www-personal.umich.edu/~mbanani/clickhere_weights/render4cnn.pth),
[Clickhere CNN weights](http://www-personal.umich.edu/~mbanani/clickhere_weights/ch_cnn.npy)).
The converted models achieve comparable performance to the Caffe for the Render For CNN model,
however, there is a larger error observed for Clickhere CNN.
<!-- We are currently training the models using PyTorch and will upload the new models soon. -->

### Running Inference

After downloading Pascal 3D+ and the pretrained weights, generating the CSV files, and setting the appropriate paths as mentioned above,
you can run inference on the Pascal 3D+ dataset by running one of the following commands (depending on the model):

        python train_keypoint.py --evaluate_only --model pretrained_clickhere --dataset pascalVehKP  
        python train_keypoint.py --evaluate_only --model pretrained_render    --dataset pascalEasy
        python train_keypoint.py --evaluate_only --model pretrained_render    --dataset pascal


#### Results

The original Render For CNN paper reported the results on the 'easy' subset of Pascal3D, which removes any truncated and occluded images from the datasets. While Click-Here CNN reports results on an augmented version of the dataset where multiple instances may belong to the same object in the image as each image-keypoint pair corresponds to an instance. Below are the results Obtained from each of the runs above:

| Accuracy (pi/6)| plane | bike   | boat  | bottle    | bus   | car   | chair | d.table   | m.bike | sofa  | train | tv    | mean  |
| ----------------------------------|:---------:|:---------:|:-----:|:---------:|:-----:|:-----:|:-----:|:-------------:|:---------:|:-----:|:-----:|:------------:|:-----:|
| R4CNN - Easy      | 80.37     | 85.59     | 62.93 | 95.60     | 94.14 | 84.08 | 82.76 | 80.95         | 85.30     | 84.61 | 84.08 | 93.26        | 84.47 |
| R4CNN - Full      | 76.26     | 69.58     | 59.03 | 87.74     | 84.32 | 69.97 | 74.2  | 66.79         | 77.29     | 82.37 | 75.48 | 81.93        | 75.41 |
| R4CNN - Veh.      | N/A       | N/A       | N/A   | N/A       | 89.26 | 74.36 | N/A   | N/A           | 81.93     | N/A   | N/A   | N/A          | 81.85 |
| CHCNN - Veh.      | N/A       | N/A       | N/A   | N/A       | 86.91 | 83.25 | N/A   | N/A           | 73.83     | N/A   | N/A   | N/A          | 81.33 |

| Median Error | plane | bike   | boat  | bottle    | bus   | car   | chair | d.table   | m.bike | sofa  | train | tv    | mean  |
| ----------------------------------|:---------:|:---------:|:-----:|:---------:|:-----:|:-----:|:-----:|:-------------:|:---------:|:-----:|:-----:|:------------:|:-----:|
| R4CNN - Easy      | 10.32     | 11.66     | 17.74 | 6.66      | 4.52  | 6.65  | 11.21 | 9.75          | 13.11     | 9.76  | 5.52  | 11.93        | 9.90  |
| R4CNN - Full      | 11.52     | 15.33     | 19.33 | 8.51      | 5.54  | 9.39  | 13.83 | 12.87         | 14.90     | 13.03 | 8.96  | 13.72        | 12.24 |
| R4CNN - Veh.      | N/A       | N/A       | N/A   | N/A       | 5.16  | 8.53  | N/A   | N/A           | 13.46     | N/A   | N/A   | N/A          | 9.05  |
| CHCNN - Veh.      | N/A       | N/A       | N/A   | N/A       | 4.01  | 8.18  | N/A   | N/A           | 19.71     | N/A   | N/A   | N/A          | 10.63 |
<!-- | ----------------------------------|:---------:|:---------:|:-----:|:---------:|:-----:|:-----:|:-----:|:-------------:|:---------:|:-----:|:-----:|:------------:|:-----:| -->

It should be noted that the results for Click-Here are lower than reported in the paper and achieved through evaluating the outputs of the author provided for the paper.
It seems that there is something incorrect with the current reimplementation and/or weight conversion. We are working on fixing this problem.


### Training the model

To train the model, simply run `python train_keypoint.py` with parameter flags as indicated in train_keypoint.py.

## Citation

This is an implementation of [Clickhere CNN](https://github.come/rszeto/click-here-cnn) and [Render For CNN](https://github.com/shapenet/RenderForCNN), so please cite the respective papers if you use this code in any published work.

## Acknowledgements

We would like to thank Ryan Szeto, Hao Su, and Charles R. Qi for providing their code, and for their assistance with questions regarding reimplementing their work. We would also like to acknowledge [Kenta Iwasaki](https://discuss.pytorch.org/u/dranithix/summary) for his advice with loss function implementation, [Pete Tae Hoon Kim](https://discuss.pytorch.org/u/thnkim/summary) for his implementation of SpatialCrossMap LRN , and [Qi Fan](https://github.com/fanq15) for releasing [caffe_to_torch_to_pytorch](https://github.com/fanq15/caffe_to_torch_to_pytorch)
