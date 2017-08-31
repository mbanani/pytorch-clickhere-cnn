# pytorch-clickhere-cnn

## Introduction

This is a [PyTorch](http://pytorch.org) implementation of [Clickhere CNN](https://github.com/rszeto/click-here-cnn) which implements the work described in the arXiv report ["Click Here: Human-Localized Keypoints as Guidance for Viewpoint Estimation"](https://arxiv.org/abs/1703.09859). We also include PyTorch models and pretrained weights for [Render for CNN project](https://github.com/shapenet/RenderForCNN). 

We currently provide the model and dataset classes, as well as the evaluation scripts. The dataset classes require the datasets used by Clickhere CNN and RenderForCNN, which makes this implementation suffer from the same storage difficulties reported by Ryan Szetor in [Clickhere CNN](https://github.come/rszeto/click-here-cnn). We will are currently revising the dataset generation and storage, and will provide a more storage-efficient implementation soon. 

There are also training scripts included which use the geometric aware loss function used by Clickhere CNN and Render For CNN. 

## Using the code

Since this is a reimplementation of [click-here-cnn](https://github.come/rszeto/click-here-cnn) and [RenderForCNN](https://github.com/shapenet/RenderForCNN), please cite the respective papers if you use this code in any published work. 

### Generating Data

We assume that you already generated the LMDB data (not the actual LMDBs) for both the Synthetic and Pascal datasets as described [here](https://github.com/rszeto/click-here-cnn/blob/master/README.md#generating-training-data). Once the LMDB data has been generated, the path in ??DEFINE?? should be set accordingly. 

### Dependencies 

Assuming that you already have the data, you need the following dependencies to run the code:
* [PyTorch 0.2](http://pytorch.org): Both PyTorch and Torch Vision
* [PyCrayon](https://github.com/torrvision/crayon): To view learning progress on TensorBoard
* [PIL](https://pypi.python.org/pypi/Pillow/2.2.1)
* [pandas](http://pandas.pydata.org/) 

## 

## Acknowledgements

I would like to thank Ryan Szeto, Hao Su, and Charles R. Qi for providing their code, and for their assistance with questions I had when I was reimplmeneting their work. 
