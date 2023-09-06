# KNOWN: Body Knowledge and Uncertainty Modeling for Monocular 3D Human Body Reconstruction
  **Body Knowledge and Uncertainty Modeling for Monocular 3D Human Body Reconstruction** <br />
  [Yufei Zhang](zhangy76@rpi.edu), Hanjing Wang, Jeffrey O. Kephart, Qiang Ji <br /> 
  ICCV2023, [arXiv](https://aps.arxiv.org/abs/2308.00799) <br />

This repository includes the Pytorch demo code and trained models for the paper. 

## Installation
```bash
conda create -n known python=3.8
conda activate known
pip install -r requirements.txt
```

## Model and Data Download
Please download the trained model and required data from [data](https://www.dropbox.com/scl/fo/12m5bz2vi0sa6exap3cmu/h?rlkey=v79f2jdfc9qgfjeywe7i7bztn&dl=0) and store them under the current directory as 
```
./data
├── cam_spin.npy
├── pose_spin.npy
├── shape_spin.npy
├── sigma_all.npy
├── model.pt
├── processed_basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
```

## Demo
We provide the demo to perform 3D reconstruction and visualize the 3D vertex prediction uncertainty (epistemic uncertainty).
```bash
python demo_img.py --img_path 'path to a testing image'
python demo_live_video.py --video_path 'empty for camera or path to a testing video'
```

## Citation
If you find our work useful for your project, please consider citing the paper:
```bibtex
@article{zhang2023body,
  title={Body Knowledge and Uncertainty Modeling for Monocular 3D Human Body Reconstruction},
  author={Zhang, Yufei and Wang, Hanjing and Kephart, Jeffrey O and Ji, Qiang},
  journal={arXiv preprint arXiv:2308.00799},
  year={2023}
}
```
