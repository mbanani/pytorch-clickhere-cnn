# pytorch-clickhere-cnn

## To Do

+ [] Upload converted weights
+ [] Test inference
+ [] Test training

## Introduction

This is a [PyTorch](http://pytorch.org) implementation of [Clickhere CNN](https://github.com/rszeto/click-here-cnn) and [Render for CNN](https://github.com/shapenet/RenderForCNN).

We currently provide the model, converted weights, dataset classes, and training/evaluation scripts. This implementation also includes an implementation of the Geometric Structure Aware loss function first mentioned in Render For CNN. The dataset classes require the datasets used by Clickhere CNN and RenderForCNN, which makes this implementation suffer from the same storage difficulties reported in [Clickhere CNN](https://github.come/rszeto/click-here-cnn). We are currently revising the dataset generation and storage, and will provide a more storage-efficient implementation soon.

If you have any questions, please email me at mbanani@umich.edu.

## Running the code 

### Generating Data

We assume that you already generated the LMDB data (not the actual LMDBs) for both the Synthetic and Pascal datasets as described [here](https://github.com/rszeto/click-here-cnn/blob/master/README.md#generating-training-data). Once the LMDB data has been generated, the path to the lmdb_data folder should be set in util.py.

### Pretrained weights

Please download the [Clickhere CNN weights](none) and [Render For CNN weights](none), and then set the appropriate paths in util.py.

### Dependencies

Assuming that you already have the data, you need the following dependencies to run the code:
* [PyTorch 0.2](http://pytorch.org) - Both PyTorch and Torch Vision
* [SciPy](https://www.scipy.org/)
* [pandas](http://pandas.pydata.org/)


### Performing Inference 

After getting the data, downloading the weights, and setting the paths in util.py, you can run inference using Render For CNN and Clickhere CNN on the 3 classes (bus, car, and motorbike) by running the following command

         python train_viewpoint.py --model pretrained_render --dataset pascal --evaluate_only 
         python train_viewpoint.py --model pretrained_clickhere --dataset pascal --evaluate_only 
 

## Citation

This is an implementation of [Clickhere CNN](https://github.come/rszeto/click-here-cnn) and [Render For CNN](https://github.com/shapenet/RenderForCNN), so please cite the respective papers if you use this code in any published work.

## Acknowledgements

I would like to thank Ryan Szeto, Hao Su, and Charles R. Qi for providing their code, and for their assistance with questions I had when I was reimplementing their work. I would also like to acknowledge [Kenta Iwasaki](https://discuss.pytorch.org/u/dranithix/summary) for his advice with loss function implementation, (Pete Tae Hoon Kim)[https://discuss.pytorch.org/u/thnkim/summary] for his implementation of SpatialCrossMap LRN , and [Qi Fan](https://github.com/fanq15) for releasing [caffe_to_torch_to_pytorch](https://github.com/fanq15/caffe_to_torch_to_pytorch)
