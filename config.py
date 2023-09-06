import torchvision.transforms as transforms

# necessary file path
SMPL_MODEL_PATH = './data/processed_basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'

SMPL_mean_cam   = './data/cam_spin.npy'
SMPL_mean_pose  = './data/pose_spin.npy'
SMPL_mean_shape = './data/shape_spin.npy'
SMPL_mean_sigma = './data/sigma_all.npy'

# image preprocessing parameters
IMG_RES = 224
IMG_RES_detect = 416
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]
trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(IMG_NORM_MEAN, IMG_NORM_STD)
])
trans_bbox = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(IMG_RES_detect),
    transforms.ToTensor()
])

# rendering parameters
FOCAL_LENGTH = 5000.
IMG_CENTER = [112,112]

# uncertainty quantification parameters
num_samples = 50


