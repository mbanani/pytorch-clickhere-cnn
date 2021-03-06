# pytorch-clickhere-cnn


## Introduction

This is a [PyTorch](http://pytorch.org) implementation of [Clickhere CNN](https://github.com/rszeto/click-here-cnn)
and [Render for CNN](https://github.com/shapenet/RenderForCNN).

We currently provide the model, converted weights, dataset classes, and training/evaluation scripts.
This implementation also includes an implementation of the Geometric Structure Aware loss function first mentioned in Render For CNN.


If you have any questions, please email me at mbanani@umich.edu.

## Getting Started

Add the repository to your python path:

        export PYTHONPATH=$PYTHONPATH:$(pwd)


Please be aware that I make use of the following packages:
- Python 3.6 
- PyTorch 1.1 and Torch Vision 0.2.2
- scipy 
- pandas 

### Generating the data
Download the [Pascal 3D+ dataset](http://cvgl.stanford.edu/projects/pascal3d.html) (Release 1.1).
Set the path for the Pascal3D directory in util/Paths.py. Finally, run the following command from the repository's root directory.

        cd data/
        python generate_pascal3d_csv.py

Please note that this will generate the csv files for 3 variants of the dataset: Pascal 3D+ (full), Pascal 3D+ (easy), and Pascal 3D-Vehicles (with keypoints). Those datasets are needed to obtain different sets of results.
Alternatively, you could directly download the csv files by using `data/get_csv.sh`

### Pre-trained Model Weights

We have converted the RenderForCNN and Clickhere model weights from the respective caffe models.
The converted models are available for download by running the script in `model_weights/get_weights.sh` 
The converted models achieve comparable performance to the Caffe for the Render For CNN
model, however, there is a larger error observed for Clickhere CNN.
Updated results coming soon.

### Running Inference

After downloading Pascal 3D+ and the pretrained weights, generating the CSV files, and setting the appropriate paths as mentioned above,
you can run inference on the Pascal 3D+ dataset by running one of the following commands (depending on the model):

        python train.py  --model chcnn --dataset pascalVehKP  
        python train.py  --model r4cnn --dataset pascalEasy
        python train.py  --model r4cnn --dataset pascal


#### Results

To be updated soon! 

The original Render For CNN paper reported the results on the 'easy' subset of Pascal3D, which removes any truncated and occluded images from the datasets. While Click-Here CNN reports results on an augmented version of the dataset where multiple instances may belong to the same object in the image as each image-keypoint pair corresponds to an instance. Below are the results Obtained from each of the runs above.

### Render For CNN paper results:

We evaluate the converted model on Pascal3D-easy as reported in the original Render For CNN paper,
as well as the full Pascal 3D dataset.
It is worth nothing that the converted model actually exceeds the performance reported in Render For CNN.

# #### Accuracy
ataset    | plane | bike  | boat  | bottle| bus   | car   | chair |d.table| mbike | sofa  | train | tv    | mean  |
|:---------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Full      | 76.26 | 69.58 | 59.03 | 87.74 | 84.32 | 69.97 | 74.2  | 66.79 | 77.29 | 82.37 | 75.48 | 81.93 | 75.41 |
| Easy      | 80.37 | 85.59 | 62.93 | 95.60 | 94.14 | 84.08 | 82.76 | 80.95 | 85.30 | 84.61 | 84.08 | 93.26 | 84.47 |
| Reported  | 74    | 83    | 52    | 91    | 91    | 88    | 86    | 73    | 78    | 90    | 86    | 92    | 82    |

#### Median Error
|dataset    | plane | bike  | boat  | bottle| bus   | car   | chair |d.table| mbike | sofa  | train | tv    | mean  |
|:---------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|Full       | 11.52 | 15.33 | 19.33 | 8.51  | 5.54  | 9.39  | 13.83 | 12.87 | 14.90 | 13.03 | 8.96  | 13.72 | 12.24 |
|Easy       | 10.32 | 11.66 | 17.74 | 6.66  | 4.52  | 6.65  | 11.21 | 9.75  | 13.11 | 9.76  | 5.52  | 11.93 | 9.90  |
|Reported   | 15.4  | 14.8  | 25.6  | 9.3   | 3.6   | 6.0   | 9.7   | 10.8  | 16.7  | 9.5   | 6.1   | 12.6  | 11.7  |



### Pascal3D - Vehicles with Keypoints

We evaluated the converted Render For CNN and Click-Here CNN models on Pascal3D-Vehicle.
It should be noted that the results for Click-Here are lower than those achieved by running the author provided Caffe code.
It seems that there is something incorrect with the current reimplementation and/or weight conversion.
We are working on fixing this problem.

#### Accuracy
|                           |  bus  | car   | m.bike | mean  |
|:-------------------------:|:-----:|:-----:|:------:|:-----:|
| Render For CNN            | 89.26 | 74.36 | 81.93  | 81.85 |
| Click-Here CNN            | 86.91 | 83.25 | 73.83  | 81.33 |
| Click-Here CNN (reported) | 96.8  | 90.2  | 85.2   | 90.7  |

#### Median Error
|                           |  bus  | car   | m.bike | mean  |
|:-------------------------:|:-----:|:-----:|:------:|:-----:|
| Render For CNN            | 5.16  | 8.53  | 13.46  | 9.05  |
| Click-Here CNN            | 4.01  | 8.18  | 19.71  | 10.63 |
| Click-Here CNN (reported) | 2.63  | 4.98  | 11.4   | 6.35  |


### Pascal3D - Vehicles with Keypoints (Fine-tuned Models)

We fine-tuned both models on the Pascal 3D+ (Vehicles with Keypoints) dataset.
Since we suspect that the problem with the replication of the Click-Here CNN model
is in the attention section, we conducted an experiment where we only fine-tuned
those weights. As reported below, fine-tuning just the attention model achieves the best performance.

|                               |  bus  | car   | m.bike | mean  |
|:-----------------------------:|:-----:|:-----:|:------:|:-----:|
| Render For CNN FT             | 93.55 | 83.98 | 87.30  | 88.28 |
| Click-Here CNN FT             | 92.97 | 89.84 | 81.25  | 88.02 |
| Click-Here CNN FT-Attention   | 94.48 | 90.77 | 84.91  | 90.05 |
| Click-Here CNN (reported)     | 96.8  | 90.2  | 85.2   | 90.7  |


#### Median Error

|                               |  bus  | car   | m.bike | mean  |
|:-----------------------------:|:-----:|:-----:|:------:|:-----:|
| Render For CNN FT             | 3.04  | 5.83  | 11.95  | 6.94  |
| Click-Here CNN FT             | 2.93  | 5.14  | 13.42  | 7.16  |
| Click-Here CNN FT-Attention   | 2.88  | 5.24  | 12.10  | 6.74  |
| Click-Here CNN (reported)     | 2.63  | 4.98  | 11.4   | 6.35  |


## Training the model

To train the model, simply run `python train.py` with parameter flags as indicated in train.py.

## Citation

This is an implementation of [Clickhere CNN](https://github.come/rszeto/click-here-cnn) and [Render For CNN](https://github.com/shapenet/RenderForCNN), so please cite the respective papers if you use this code in any published work.

## Acknowledgements

We would like to thank Ryan Szeto, Hao Su, and Charles R. Qi for providing their code, and
for their assistance with questions regarding reimplementing their work. We would also
like to acknowledge [Kenta Iwasaki](https://discuss.pytorch.org/u/dranithix/summary) for
his advice with loss function implementation and [Qi Fan](https://github.com/fanq15) for releasing
[caffe_to_torch_to_pytorch](https://github.com/fanq15/caffe_to_torch_to_pytorch).

This work has been partially supported by DARPA W32P4Q-15-C-0070 (subcontract from SoarTech) and funds from the University of Michigan Mobility Transformation Center.
