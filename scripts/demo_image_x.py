import argparse
import os
import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm

from hybrik.models import builder
from hybrik.utils.config import update_config
from hybrik.utils.presets import SimpleTransform3DSMPLCam
from hybrik.utils.render_pytorch3d import render_mesh
from hybrik.utils.vis import get_one_box, vis_2d

det_transform = T.Compose([T.ToTensor()])


def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    return [cx, cy, w, h]


parser = argparse.ArgumentParser(description='HybrIK Image Demo')

parser.add_argument('--gpu', default=0, type=int, help='GPU ID')
parser.add_argument('--img-dir', required=True, type=str, help='Path to directory with images')
parser.add_argument('--out-dir', required=True, type=str, help='Output folder')
parser.add_argument('--save-img', action='store_true', help='Save processed images')

opt = parser.parse_args()

cfg_file = 'configs/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml'
CKPT = './pretrained_models/hybrik_hrnet.pth'
cfg = update_config(cfg_file)

bbox_3d_shape = [item * 1e-3 for item in getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))]

dummpy_set = edict({'bbox_3d_shape': bbox_3d_shape})

transformation = SimpleTransform3DSMPLCam(
    dummpy_set, scale_factor=cfg.DATASET.SCALE_FACTOR,
    color_factor=cfg.DATASET.COLOR_FACTOR, occlusion=cfg.DATASET.OCCLUSION,
    input_size=cfg.MODEL.IMAGE_SIZE, output_size=cfg.MODEL.HEATMAP_SIZE,
    depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM, bbox_3d_shape=bbox_3d_shape,
    rot=cfg.DATASET.ROT_FACTOR, sigma=cfg.MODEL.EXTRA.SIGMA,
    train=False, add_dpg=False, loss_type=cfg.LOSS['TYPE'])

det_model = fasterrcnn_resnet50_fpn(pretrained=True)
hybrik_model = builder.build_sppe(cfg.MODEL)

print(f'Loading model from {CKPT}...')
hybrik_model.load_state_dict(torch.load(CKPT, map_location='cpu')['model'])

det_model.cuda(opt.gpu).eval()
hybrik_model.cuda(opt.gpu).eval()

os.makedirs(opt.out_dir, exist_ok=True)

img_files = sorted([f for f in os.listdir(opt.img_dir) if f.endswith(('.jpg', '.png'))])

for img_file in tqdm(img_files):
    img_path = os.path.join(opt.img_dir, img_file)
    input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        det_input = det_transform(input_image).to(opt.gpu)
        det_output = det_model([det_input])[0]

        tight_bbox = get_one_box(det_output)
        if tight_bbox is None:
            continue

        pose_input, bbox, img_center = transformation.test_transform(input_image, tight_bbox)
        pose_input = pose_input.to(opt.gpu)[None, :, :, :]
        pose_output = hybrik_model(
            pose_input, flip_test=True,
            bboxes=torch.from_numpy(np.array(bbox)).to(pose_input.device).unsqueeze(0).float(),
            img_center=torch.from_numpy(img_center).to(pose_input.device).unsqueeze(0).float()
        )

        uv_29 = pose_output.pred_uvd_jts.reshape(29, 3)[:, :2]
        transl = pose_output.transl.detach()
        vertices = pose_output.pred_vertices.detach()

        smpl_faces = torch.from_numpy(hybrik_model.smpl.faces.astype(np.int32))
        focal = 1000.0
        bbox_xywh = xyxy2xywh(bbox)
        transl_camsys = transl.clone() * 256 / bbox_xywh[2]
        focal = focal / 256 * bbox_xywh[2]

        color = render_mesh(vertices=vertices, faces=smpl_faces, translation=transl_camsys,
                            focal_length=focal, height=input_image.shape[0], width=input_image.shape[1])

        valid_mask = (color[:, :, :, [-1]] > 0)
        image_vis = (color[:, :, :, :3] * valid_mask + (1 - valid_mask) * input_image).cpu().numpy()[0]

        image_vis = cv2.cvtColor((image_vis * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        if opt.save_img:
            cv2.imwrite(os.path.join(opt.out_dir, f'processed_{img_file}'), image_vis)

print("Processing complete!")
