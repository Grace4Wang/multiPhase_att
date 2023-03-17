# MODEL_WEIGHT="Path/of/Model/Weight/XXX.pth"
MODEL_WEIGHT = "E:/LSX\Bayer/bbox_ing/aug_model/weights/1_1_AP.pth"
CLASS_INDEX=1 # Index of the class for which you want to see the Guided-gradcam
INPUT_PATCH_SIZE_SLICE_NUMBER=16 # Input patch slice you want to feed at a time
LAYER_NAME='conv3' # Name of the layer from where you want to get the Guided-GradCAM
# LAYER_NAME = [1,2,3,4]
NIFTI_PATH="E:/Bayer/data/val/AP/003.nii.gz"
SAVE_PATH="E:/Bayer/data/figures/AP/003.nii.gz"
