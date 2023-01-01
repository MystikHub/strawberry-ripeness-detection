# Implementation A

This directory contains the code for the Mask R-CNN implementation of our
strawberry detector. The code for training and inference with the assignment
dataset can be found in the `strawberry.py` file.

## Setup

To set up the project:

1. Install Python version 3.10.8 or newer
2. Install the Mask R-CNN dependencies with:
    - `pip install -r mask-rcnn
3. If you would like to train the model on an Nvidia GPU, additionally install:
    - `tensorflow-gpu>=2.11.0` (and optionally uninstall the `tensorflow`
      package). This step may not be necessary, since tensorflow might have
      built-in GPU support, though we have not tested this
    - `tensorrt>=8.5.2.2`
    - Depending on your setup, you may also need to install Nvidia's CUDA tools.
      We verified that our code worked with CUDA version 11.8.0 and cuDNN 8.6.0
4. Download the strawberry dataset into `repo/data` with these directories:
    - `repo/data/bounding_box` containing the bounding box text files
    - `repo/data/images/training` containing the strawberry photographs used for
      training (we used 80% of the original dataset)
    - `repo/data/images/validation` containing the strawberry photographs used
      for validation (we used 20% of the original dataset)
    - (Optional) `repo/data/instance_segmentation` containing the instance
      segmentation data files. These files are not used during training or
      evaluation since the same information is available in the
      `instance+ripeness_segmantation` files
    - `repo/data/instance+ripeness_segmentation` containing the instance and
      ripeness segmentation data files
5. Download the COCO pre-trained weights (`mask_rcnn_coco.h5`) from the [Mask
   R-CNN 2.0 release on
   GitHub](https://github.com/matterport/Mask_RCNN/releases/tag/v2.0) and save
   it as `repo/implementation-a/mask_rcnn_coco.h5`

