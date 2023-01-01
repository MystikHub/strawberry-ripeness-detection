"""
Mask R-CNN
Train on the strawberry dataset and detect strawberries and their ripeness

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 strawberry.py train --dataset=/path/to/strawberry/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 strawberry.py train --dataset=/path/to/strawberry/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 strawberry.py train --dataset=/path/to/strawberry/dataset --weights=imagenet

    # Detect strawberries in an image
    python3 strawberry.py detect --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 strawberry.py splash --weights=last --video=<URL or path to file>
"""

import cv2
import datetime
import json
import numpy as np
import os
import random
import skimage.draw
import sys

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import model as modellib, utils
from mrcnn import visualize
from mrcnn.config import Config
from mrcnn.model import log

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class StrawberryConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "strawberry"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    # IMAGES_PER_GPU = 2
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # Background + three strawberry classes

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # USE_MINI_MASK = False


############################################################
#  Dataset
############################################################

class StrawberryDataset(utils.Dataset):

    def load_strawberry(self, dataset_dir, subset):
        """Load a subset of the strawberry dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes
        self.add_class("strawberry", 1, "Unripe")
        self.add_class("strawberry", 2, "Partially ripe")
        self.add_class("strawberry", 3, "Ripe")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        # dataset_dir = os.path.join(dataset_dir, subset)
        images_dir = os.path.join(dataset_dir, "images")

        image_file_names = os.listdir(images_dir)

        # Add images
        for image_file_name in image_file_names:

            # Get rid of the ".txt"
            image_name = image_file_name[:-4]
            image_path = os.path.join(images_dir, image_file_name)

            self.add_image(
                "strawberry",
                image_id=image_name,  # use file name as a unique image id
                path=image_path,
                width=1008, height=756)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        
        image_info = self.image_info[image_id]
        image_name = image_info["id"]
        print("Reading strawberry {}".format(image_name))

        # # If not a strawberry dataset image, delegate to parent class.
        if image_info["source"] != "strawberry":
            return super(self.__class__, self).load_mask(image_id)
        
        instance_image_path = os.path.join(args.dataset, "instance_segmentation", str(image_name)) + ".png"
        instance_ripeness_image_path = os.path.join(args.dataset, "instance+ripeness_segmentation", str(image_name)) + ".png"
        instance_image = np.array(cv2.imread(instance_image_path))[:, :, 0]
        instance_ripeness_image = cv2.imread(instance_image_path)
        bb_data = np.loadtxt(os.path.join(args.dataset, "bounding_box", str(image_name)) + ".txt")
        class_id_to_channel = [1, 2, 0]
        
        bb_data_rows = 1
        if len(bb_data.shape) == 2:
            bb_data_rows = bb_data.shape[0]

        mask = np.zeros([image_info["height"], image_info["width"], bb_data_rows])
        class_ids = []
        for i in range(bb_data_rows):
            class_id = -1
            if len(bb_data.shape) == 1:
                class_id = int(bb_data[0])
            elif len(bb_data.shape) == 2:
                class_id = int(bb_data[i][0])

            class_ids.append(class_id + 1)

            mask[:, :, i] = np.where(instance_image == i + 1, 1, 0)
        
        return mask.astype(np.uint8), np.array(class_ids).astype(np.int32)

        # # Convert polygons to a bitmap mask of shape
        # # [height, width, instance_count]
        # info = self.image_info[image_id]
        # mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
        #                 dtype=np.uint8)
        # for i, p in enumerate(info["polygons"]):
        #     # Get indexes of pixels inside the polygon and set them to 1
        #     rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
        #     mask[rr, cc, i] = 1

        # # Return mask, and array of class IDs of each instance. Since we have
        # # one class ID only, we return an array of 1s
        # return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        return self.image_info.path
        # info = self.image_info[image_id]
        # if info["source"] == "strawberry":
        #     return info["path"]
        # else:
        #     super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = StrawberryDataset()
    dataset_train.load_strawberry(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = StrawberryDataset()
    dataset_val.load_strawberry(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                # epochs=30,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)

def detect_strawberry(model, inference_config, image_path):
    
    dataset_train = StrawberryDataset()
    dataset_train.load_strawberry(args.dataset, "train")
    dataset_train.prepare()

    image_id = random.choice(dataset_train.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_train, inference_config, 
                            image_id)
    
    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
        dataset_train.class_names, figsize=(8, 8))

    # results = model.detect([original_image], verbose=1)

    # r = results[0]
    # visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
    #                             dataset_train.class_names, r['scores'], ax=get_ax())


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect strawberries and their ripeness.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/strawberry/dataset/",
                        help='Directory of the strawberry dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.image, "Provide --image to detect strawberries"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = StrawberryConfig()
    else:
        class InferenceConfig(StrawberryConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            USE_MINI_MASK = False
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "detect":
        detect_strawberry(model, config, image_path=args.image)
        # detect_and_color_splash(model, image_path=args.image,
        #                         video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))