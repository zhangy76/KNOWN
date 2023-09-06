import os
import sys
import time
import argparse
sys.path.append('../')
from yolov3.yolo import YOLOv3

import config

import cv2
import numpy as np
import trimesh
import torch

from utils import perspective_projection, crop2expandsquare_zeros, rot6d_to_rotmat
from renderer import Renderer

from hmr import hmr
from smpl_torch import SMPL


def main(img_path, save_img=False, checkpoint='./data/model.pt'):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    detector = YOLOv3(device=device, img_size=config.IMG_RES_detect, person_detector=True)
    print('yolo human detector loaded')

    model = hmr()
    model.load_state_dict(torch.load(checkpoint), strict=False)
    model.to(device)
    model.eval()
    print('KNOWN model loaded')

    smpl = SMPL(config.SMPL_MODEL_PATH, device)
    faces_txt = torch.tensor(smpl.faces.astype(np.int)).unsqueeze(0).to(device)
    print('SMPL model loaded')

    frame = cv2.imread(img_path)
    H_frame, W_frame, _ = frame.shape

    start_time = time.time()

    if W_frame>H_frame:
        frame, _, Len_max_ori = crop2expandsquare_zeros(frame, [int((W_frame-H_frame)/2), 0, int((W_frame-H_frame)/2)+H_frame, H_frame])
    else:
        frame, _, Len_max_ori = crop2expandsquare_zeros(frame, [0, int((H_frame-W_frame)/2), W_frame, int((H_frame-W_frame)/2) + W_frame])
    frame_detect = cv2.resize(frame, (config.IMG_RES_detect, config.IMG_RES_detect))

    # preprocess the image and obtain human bounding box
    img = config.trans_bbox(frame_detect[:, :, ::-1]).float().unsqueeze(0).to(device)
    detections = detector(img)

    frame_output = np.ones([config.IMG_RES_detect, config.IMG_RES_detect*3, 3]) 
    if len(detections) < 1 or len(detections[0])==3 or detections[0].shape[0] == 0:
        fps = "%.2f fps" % (1 / (time.time()-start_time))
        frame_output[:, :config.IMG_RES_detect] = frame_detect / 255.
        frame_output[:, config.IMG_RES_detect:config.IMG_RES_detect*2] = frame_detect / 255.
        cv2.putText(frame_output, fps, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        if save_img:
            cv2.imwrite('output.jpg', np.uint8(frame_output*255))

        print('failed to detect a person from the input image')
        cv2.imshow('Video Demo', frame_output)
        if cv2.waitKey(0) & 0xff == 27: # exit if pressed `ESC`
            cv2.destroyAllWindows()

    x1, y1, x2, y2, conf, cls_conf, cls_pred = detections[0][0]
    center = [(x1+x2)/2, (y1+y2)/2]
    scalex = 1.3
    scaley = 1.1
    x1 = max(int(center[0] + (x1-center[0])*scalex), 0)
    y1 = max(int(center[1] + (y1-center[1])*scaley), 0)
    x2 = min(int(center[0] + (x2-center[0])*scalex), config.IMG_RES_detect)
    y2 = min(int(center[1] + (y2-center[1])*scaley), config.IMG_RES_detect)

    bbox = [int(x1), int(y1), int(x2), int(y2)]

    # image preprocessing
    X_cropped, center, Len_max = crop2expandsquare_zeros(frame_detect, bbox)
    X_cropped = cv2.resize(X_cropped, (config.IMG_RES, config.IMG_RES), interpolation = cv2.INTER_CUBIC)
    X = config.trans(X_cropped).float().unsqueeze(0).to(device)

    # prediction
    pred_pose_mean, pred_beta_mean, pred_cam_mean, pred_pose_sigma, pred_beta_sigma, pred_kp_sigma = model(X)

    # uncertainty quantification
    epsilon = torch.normal(0, 1, size=(config.num_samples, 138+10+46), device=device)
    pred_pose_sample = pred_pose_mean.expand(config.num_samples, -1).clone()
    pred_pose_sigma[:,36:48] = 0
    pred_pose_sigma[:,54:66] = 0
    pred_pose_sigma[:,114:138] = 0 # ignore them for visualization purpose

    pred_pose_sample[:,6:] = pred_pose_sample[:,6:].clone() + epsilon[:,:138]*pred_pose_sigma
    pred_pose_sample_rotmat = rot6d_to_rotmat(pred_pose_sample).view(config.num_samples, 24, 3, 3)

    pred_beta_sample = pred_beta_mean.clone() + epsilon[:, 138:148]*pred_beta_sigma
    pred_output_sample = smpl.forward(betas=pred_beta_sample, thetas_rotmat=pred_pose_sample_rotmat)
    pred_vertices_sample = pred_output_sample.vertices.detach().cpu().numpy()

    uncertainty_3D = np.linalg.norm(pred_vertices_sample-np.mean(pred_vertices_sample,axis=0, keepdims=True), axis=2).mean(axis=0).copy()
    uncertainty_3D_color = trimesh.visual.interpolate(uncertainty_3D.copy(), 'Spectral')[:, :3]

    # inference
    focal_length = config.FOCAL_LENGTH 
    pred_rotmat = rot6d_to_rotmat(pred_pose_mean).view(1, 24, 3, 3)
    jidx = [7,8,10,11,20,21,22,23] # ingore these joints due to lack of supervision
    pred_rotmat[:,jidx] = torch.eye(3, device=device).unsqueeze(0)
    pred_output = smpl.forward(betas=pred_beta_mean, thetas_rotmat=pred_rotmat)
    pred_vertices = pred_output.vertices
    pred_cam_t = torch.stack([pred_cam_mean[:,1],
                                  pred_cam_mean[:,2],
                                  focal_length/(pred_cam_mean[:,0] +1e-9)],dim=-1)
    pred_vertices = pred_vertices + pred_cam_t

    # set up the renderer
    renderer = Renderer(focal_length=focal_length*Len_max/config.IMG_RES, img_res=Len_max, faces=smpl.faces)

    # obtain the render image
    bbox_local = [bbox[0]-int(center[0]-Len_max//2), bbox[1]-int(center[1]-Len_max//2),
                      bbox[2]-int(center[0]-Len_max//2), bbox[3]-int(center[1]-Len_max//2)]
    rend_body, rend_depth = renderer.render(pred_vertices[0].detach().cpu().numpy(), uncertainty_3D_color, 0, 0, 0)
    rend_body = rend_body[bbox_local[1]:bbox_local[3],bbox_local[0]:bbox_local[2]]
    rend_depth = rend_depth[bbox_local[1]:bbox_local[3],bbox_local[0]:bbox_local[2]]
    rend_mask = (rend_depth>0)[:,:,None] * 1

    # overlap the render image with the input frame
    frame_overlap = frame_detect.copy()
    frame_overlap[bbox[1]:bbox[3],bbox[0]:bbox[2]] = (1-rend_mask)*frame_overlap[bbox[1]:bbox[3],bbox[0]:bbox[2]] +\
                                                                        rend_mask*rend_body[:, :, :3]

    renderer = Renderer(focal_length=focal_length*config.IMG_RES_detect / config.IMG_RES / 2, img_res=config.IMG_RES_detect, faces=smpl.faces)
    rend_body_ori, _ = renderer.render(pred_vertices[0].detach().cpu().numpy(), uncertainty_3D_color, 0, 0, 0)

    frame_output = np.concatenate([frame_detect, frame_overlap, rend_body_ori[:,:,:3]], axis=1) / 255.
    fps = "%.2f fps" % (1 / (time.time()-start_time))
    cv2.rectangle(frame_output, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (255,0,0), 2)
    cv2.putText(frame_output, fps, (5,50 ), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    if save_img:
        cv2.imwrite('output.jpg', np.uint8(frame_output*255))

    cv2.imshow('Video Demo', frame_output)
    if cv2.waitKey(0) & 0xff == 27: # exit if pressed `ESC`
        cv2.destroyAllWindows()

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default='', help='Path to the image')
parser.add_argument('--save', type=bool, default=False, help='Whether saving the results or not')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')

if __name__ == '__main__':
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main(args.img_path, args.save)
