'''-----------------------------CIFAR10 Datset properties------------------------------'''
CIFAR_10_DATASET_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_10_DATASET_STANDARD_DEVIATION = [0.2470, 0.2435, 0.2616]

'''--------------------Data Augmentation (Albumentation) properties--------------------'''

#Horizontal Flip
AUGMENTATION_HORIZONTAL_FLIP_PROB = 0.25

#Cutout
AUGMENTATION_CUTOUT_MAX_HOLES = 1
AUGMENTATION_CUTOUT_MIN_HOLES = 1
AUGMENTATION_CUTOUT_PROB = 0.5
AUGMENTATION_CUTOUT_MAX_HEIGHT = 16  # 32/2
AUGMENTATION_CUTOUT_MAX_WIDTH = 16  # 32/2
AUGMENTATION_CUTOUT_MIN_HEIGHT = 16  # 32/2
AUGMENTATION_CUTOUT_MIN_WIDTH = 16  # 32/2

#Padding
AUGMENTATION_PADDING_MIN_HEIGHT = 36
AUGMENTATION_PADDING_MIN_WIDTH = 36

#Random Crop
AUGMENTATION_RANDOM_CROP_WIDTH = 32
AUGMENTATION_RANDOM_CROP_HEIGHT = 32
AUGMENTATION_RANDOM_CROP_ALWAYS_APPLY = False

#Mask Fill Value
AUGMENTATION_MASK_FILL_VALUE = None