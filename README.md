# pytorch-clickhere-cnn

## Introduction

This is a [PyTorch](http://pytorch.org) implementation of [Clickhere CNN](https://github.com/rszeto/click-here-cnn) and [Render for CNN](https://github.com/shapenet/RenderForCNN).

We currently provide the model, converted weights, dataset classes, and training/evaluation scripts. This implementation also includes an implementation of the Geometric Structure Aware loss function first mentioned in Render For CNN. The dataset classes require the datasets used by Clickhere CNN and RenderForCNN, which makes this implementation suffer from the same storage difficulties reported in [Clickhere CNN](https://github.come/rszeto/click-here-cnn). We are currently revising the dataset generation and storage, and will provide a more storage-efficient implementation soon.

If you have any questions, please email me at mbanani@umich.edu.

## Running the code

### Generating Data

The LMDB data (not the actual LMDBs) have to be generated for both the Synthetic and Pascal datasets as described [here](https://github.com/rszeto/click-here-cnn/blob/master/README.md#generating-training-data). Once the LMDB data has been generated, the path to the lmdb_data folder should be set in util.py.

Note: When generating the data, please edit get_job_key() function in generate_lmdb_data.py as shown below. This edit allows the lmdb keys to have information about the bounding box, which allows us to know why object in the image does the key_point belong to.

    def get_job_key(job):
        key_prefix, line, reverse = job

        # Extract image path from the line
        m = re.match(utils.LINE_FORMAT, line)
        full_image_path = m.group(1)
        bb0 = m.group(2)
        bb1 = m.group(3)
        bb2 = m.group(4)
        bb3 = m.group(5)
        keypoint_class = m.group(8)
        obj_class_id = m.group(9)
        # obj_id = m.group(13)
        obj_id = '0'
        # Get the file name without the extension
        full_image_name, _ = os.path.splitext(os.path.basename(full_image_path))
        if reverse:
            return key_prefix + '_' + full_image_name + '_obj' + obj_id + '_objc' + obj_class_id + '_bbox' + bb0 + '-' + bb1 + '-' + bb2 + '-' + bb3 + '_kp' + keypoint_class + '_r'
        else:
            return key_prefix + '_' + full_image_name + '_obj' + obj_id + '_objc' + obj_class_id + '_bbox' + bb0 + '-' + bb1 + '-' + bb2 + '-' + bb3 + '_kp' + keypoint_class


### Pretrained weights

We have converted the RenderForCNN and Clickhere model weights from caffe models, which can be downloaded here ([Render For CNN weights](umich.edu/~mbanani/clickhere_weights/render4cnn.pth), [Clickhere CNN weights](umich.edu/~mbanani/clickhere_weights/ch_cnn.npy)). The converted models achieve comparable performance to the Caffe for the Render For CNN model, however, there is a larger error observed for Clickhere CNN. We are currently training the models using PyTorch, and will upload the new models soon.

### Dependencies

Assuming that you already have the data, you need the following dependencies to run the code:
* [PyTorch 0.2](http://pytorch.org) - Both PyTorch and Torch Vision
* [SciPy](https://www.scipy.org/)
* [pandas](http://pandas.pydata.org/)
* [PyCrayon](https://github.com/torrvision/crayon)


### Performing Inference

After getting the data, downloading the weights, and setting the paths in util.py, you can run inference using Render For CNN and Clickhere CNN on the 3 classes (bus, car, and motorbike) by running the following command

         python train_viewpoint.py --model pretrained_render --dataset pascal --evaluate_only
         python train_viewpoint.py --model pretrained_clickhere --dataset pascal --evaluate_only


## Citation

This is an implementation of [Clickhere CNN](https://github.come/rszeto/click-here-cnn) and [Render For CNN](https://github.com/shapenet/RenderForCNN), so please cite the respective papers if you use this code in any published work.

## Acknowledgements

I would like to thank Ryan Szeto, Hao Su, and Charles R. Qi for providing their code, and for their assistance with questions I had when I was reimplementing their work. I would also like to acknowledge [Kenta Iwasaki](https://discuss.pytorch.org/u/dranithix/summary) for his advice with loss function implementation, [Pete Tae Hoon Kim](https://discuss.pytorch.org/u/thnkim/summary) for his implementation of SpatialCrossMap LRN , and [Qi Fan](https://github.com/fanq15) for releasing [caffe_to_torch_to_pytorch](https://github.com/fanq15/caffe_to_torch_to_pytorch)
