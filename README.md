# viewpoint_estimation

## Introduction

This is a [PyTorch](http://pytorch.org) implementation of [Clickhere CNN](https://github.com/rszeto/click-here-cnn)
and [Render for CNN](https://github.com/shapenet/RenderForCNN).

We currently provide the model, converted weights, dataset classes, and training/evaluation scripts.
This implementation also includes an implementation of the Geometric Structure Aware loss function first mentioned in Render For CNN.
If you have any questions, please email me at mbanani@umich.edu.


## Citation

This is an implementation of [Clickhere CNN](https://github.come/rszeto/click-here-cnn) and [Render For CNN](https://github.com/shapenet/RenderForCNN), so please cite the respective papers if you use this code in any published work.

## Acknowledgements

We would like to thank Ryan Szeto, Hao Su, and Charles R. Qi for providing their code, and for their assistance with questions regarding reimplementing their work. We would also like to acknowledge [Kenta Iwasaki](https://discuss.pytorch.org/u/dranithix/summary) for his advice with loss function implementation, [Pete Tae Hoon Kim](https://discuss.pytorch.org/u/thnkim/summary) for his implementation of SpatialCrossMap LRN , and [Qi Fan](https://github.com/fanq15) for releasing [caffe_to_torch_to_pytorch](https://github.com/fanq15/caffe_to_torch_to_pytorch)
