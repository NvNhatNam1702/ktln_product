import sys
import os
from typing import List, Union
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import imageio
import tqdm
from pyhocon import ConfigFactory

# --- 1. SETUP PATHS ---
PIXEL_NERF_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "pixel-nerf"))
sys.path.insert(0, os.path.join(PIXEL_NERF_ROOT_DIR, "src"))

try:
    from model import make_model
    from render import NeRFRenderer
    from util import get_cuda, coord_from_blender, pose_spherical, gen_rays
except ImportError as e:
    print(f"[ERROR] Could not import from pixel-nerf/src: {e}")
    sys.exit(1)


class PixelNeRFWrapper:
    def __init__(self, checkpoint_path, conf_path, gpu_id=0):
        self.device = get_cuda(gpu_id)
        self.gpu_id = gpu_id

        # --- 2. LOAD CONFIG FROM DISK ---
        if not os.path.isabs(conf_path):
            full_conf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), conf_path))
        else:
            full_conf_path = conf_path

        if not os.path.exists(full_conf_path):
            print(f"ERROR: Config file not found at {full_conf_path}")
            sys.exit(1)

        print(f"Loading configuration from: {full_conf_path}")
        self.conf = ConfigFactory.parse_file(full_conf_path)

        # --- 3. INITIALIZE MODEL ---
        print("Initializing Model...")
        self.model = make_model(self.conf.get_config("model")).to(self.device)

        # --- 4. LOAD WEIGHTS ---
        if not os.path.isabs(checkpoint_path):
            full_ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), checkpoint_path))
        else:
            full_ckpt_path = checkpoint_path

        if os.path.isdir(full_ckpt_path):
            full_ckpt_path = os.path.join(full_ckpt_path, "pixel_nerf_latest")

        print(f"Loading weights from: {full_ckpt_path}")
        ckpt = torch.load(full_ckpt_path, map_location=self.device)

        if "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
        else:
            self.model.load_state_dict(ckpt)

        self.model.eval()

        # --- 5. INITIALIZE RENDERER ---
        self.render_batch_size = 40000
        self.renderer = NeRFRenderer.from_conf(
            self.conf.get_config("renderer"), eval_batch_size=self.render_batch_size
        ).to(self.device)
        self.render_par = self.renderer.bind_parallel(self.model, [gpu_id], simple_output=True).eval()

    def render_video(self, image_path: Union[str, List[str]], num_views=40, fps=15, radius=1.3, elevation=-30.0, gif=True):
        in_sz = 128
        out_sz = 128

        z_near = 0.8
        z_far = 1.8
        focal_val = 131.25
        focal = torch.tensor(focal_val, dtype=torch.float32, device=self.device)

        if isinstance(image_path, (str, os.PathLike)):
            image_paths = [image_path]
        else:
            image_paths = list(image_path)

        if len(image_paths) == 0:
            raise ValueError("No image paths provided to render_video")

        transform = transforms.Compose([
            transforms.Resize((in_sz, in_sz)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        image_tensors = []
        cam_poses = []
        output_root = os.path.splitext(image_paths[0])[0]

        for single_path in image_paths:
            root, ext = os.path.splitext(single_path)
            norm_image_path = f"{root}_normalize{ext}"
            target_path = norm_image_path if os.path.exists(norm_image_path) else single_path
            print(f"Processing: {target_path}")

            image = Image.open(target_path).convert("RGB")
            image_tensors.append(transform(image))

            cam_pose = torch.eye(4, device=self.device)
            cam_pose[2, -1] = radius
            cam_poses.append(cam_pose)

        image_tensor = torch.stack(image_tensors, dim=0).unsqueeze(0).to(self.device)
        cam_pose = torch.stack(cam_poses, dim=0).unsqueeze(0).to(self.device)

        _coord_from_blender = coord_from_blender()
        render_poses = torch.stack(
            [
                _coord_from_blender @ pose_spherical(angle, elevation, radius)
                for angle in np.linspace(-180, 180, num_views + 1)[:-1]
            ],
            0,
        )
        render_rays = gen_rays(render_poses, out_sz, out_sz, focal, z_near, z_far).to(self.device)

        with torch.no_grad():
            self.model.encode(
                image_tensor,
                cam_pose,
                focal
            )

            all_rgb_fine = []
            for rays in tqdm.tqdm(torch.split(render_rays.view(-1, 8), self.render_batch_size, dim=0)):
                rgb, _depth = self.render_par(rays[None])
                all_rgb_fine.append(rgb[0])

            rgb_fine = torch.cat(all_rgb_fine)
            frames = (rgb_fine.view(num_views, out_sz, out_sz, 3).cpu().numpy() * 255).astype(np.uint8)

            im_name = os.path.basename(output_root)
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)

            if gif:
                vid_path = os.path.join(output_dir, im_name + "_spin.gif")
                imageio.mimwrite(vid_path, frames, fps=fps, loop=0)
            else:
                vid_path = os.path.join(output_dir, im_name + "_spin.mp4")
                imageio.mimwrite(vid_path, frames, fps=fps, quality=8)

            print(f"Result saved: {vid_path}")

        return vid_path

