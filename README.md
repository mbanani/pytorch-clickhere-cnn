# viewpoint_estimation


## To Do

    - [x] Convert all links to relative paths!
    - [ ] Include corrected links to weights! (check!)
    - [ ] Get RenderForCNN and ClickHere Running (without regards to accuracy)
    - [ ] Test suspected modification to map generation (flipping X axis)
    - [ ] Update README to reflect current content
    - [ ] Fix Loss Function to match what was in the r4cnn paper (ignore what their code actually did for now -- avoid normalization)
    - [ ] Normalize Class Naming methodology (Capital vs normal .. )
    - [ ] Try to resolve dependence on util .. be more systematic about my imports!

## Introduction

This is a [PyTorch](http://pytorch.org) implementation of [Clickhere CNN](https://github.com/rszeto/click-here-cnn)
and [Render for CNN](https://github.com/shapenet/RenderForCNN).

We currently provide the model, converted weights, dataset classes, and training/evaluation scripts.
This implementation also includes an implementation of the Geometric Structure Aware loss function first mentioned in Render For CNN.


If you have any questions, please email me at mbanani@umich.edu.


## Getting Started

### Generating the data

### Pre-trained Model Weights

We have converted the RenderForCNN and Clickhere model weights from the respective caffe models.
The converted models can be downloaded here
([Render For CNN weights](http://www-personal.umich.edu/~mbanani/clickhere_weights/render4cnn.pth),
[Clickhere CNN weights](http://www-personal.umich.edu/~mbanani/clickhere_weights/ch_cnn.npy)).
The converted models achieve comparable performance to the Caffe for the Render For CNN model,
however, there is a larger error observed for Clickhere CNN.
We are currently training the models using PyTorch and will upload the new models soon.


### Running Inference

### Training the model (in-progress)

## Citation

This is an implementation of [Clickhere CNN](https://github.come/rszeto/click-here-cnn) and [Render For CNN](https://github.com/shapenet/RenderForCNN), so please cite the respective papers if you use this code in any published work.

## Acknowledgements

We would like to thank Ryan Szeto, Hao Su, and Charles R. Qi for providing their code, and for their assistance with questions regarding reimplementing their work. We would also like to acknowledge [Kenta Iwasaki](https://discuss.pytorch.org/u/dranithix/summary) for his advice with loss function implementation, [Pete Tae Hoon Kim](https://discuss.pytorch.org/u/thnkim/summary) for his implementation of SpatialCrossMap LRN , and [Qi Fan](https://github.com/fanq15) for releasing [caffe_to_torch_to_pytorch](https://github.com/fanq15/caffe_to_torch_to_pytorch)
