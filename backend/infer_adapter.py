import sys
import os
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
    from util import get_cuda, coord_from_blender, pose_spherical, gen_rays, gen_grid
    from util.recon import marching_cubes, save_obj
except ImportError as e:
    print(f"[ERROR] Could not import from pixel-nerf/src: {e}")
    sys.exit(1)

class PixelNeRFWrapper:
    def __init__(self, checkpoint_path, conf_path, gpu_id=0):
        self.device = get_cuda(gpu_id)
        self.gpu_id = gpu_id

        # --- 2. LOAD CONFIG FROM DISK ---
        # We use the path you found: pixel-nerf/conf/exp/...
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
        # Now we use the loaded config to build the model structure
        self.model = make_model(self.conf.get_config("model")).to(self.device)

        # --- 4. LOAD WEIGHTS ---
        if not os.path.isabs(checkpoint_path):
            full_ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), checkpoint_path))
        else:
            full_ckpt_path = checkpoint_path

        # Handle Directory Inputs
        if os.path.isdir(full_ckpt_path):
            full_ckpt_path = os.path.join(full_ckpt_path, "pixel_nerf_latest")

        print(f"Loading weights from: {full_ckpt_path}")
        ckpt = torch.load(full_ckpt_path, map_location=self.device)
        
        # Robust loading to handle different checkpoint structures
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

    def render_video(self, image_path, num_views=40, fps=15, radius=1.3, elevation=-30.0, gif=True):
        # Read resolution from config if available, else default to 128
        in_sz = 128
        out_sz = 128
        
        z_near = 0.8
        z_far = 1.8
        focal_val = 131.25
        focal = torch.tensor(focal_val, dtype=torch.float32, device=self.device)

        # Auto-append _normalize logic
        root, ext = os.path.splitext(image_path)
        norm_image_path = f"{root}_normalize{ext}"
        target_path = norm_image_path if os.path.exists(norm_image_path) else image_path
        print(f"Processing: {target_path}")

        image = Image.open(target_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((in_sz, in_sz)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        image_tensor = transform(image).unsqueeze(0).unsqueeze(0).to(self.device)

        cam_pose = torch.eye(4, device=self.device)
        cam_pose[2, -1] = radius

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
                cam_pose.unsqueeze(0).unsqueeze(0), 
                focal
            )

            all_rgb_fine = []
            for rays in tqdm.tqdm(torch.split(render_rays.view(-1, 8), self.render_batch_size, dim=0)):
                rgb, _depth = self.render_par(rays[None])
                all_rgb_fine.append(rgb[0])

            rgb_fine = torch.cat(all_rgb_fine)
            frames = (rgb_fine.view(num_views, out_sz, out_sz, 3).cpu().numpy() * 255).astype(np.uint8)

            im_name = os.path.basename(root)
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

    def extract_mesh(self, image_path, output_format="obj", resolution=128, 
                     bounds=None, isosurface=50.0, 
                     radius=1.3, focal_val=131.25, z_near=0.8, z_far=1.8):
        """
        Extract a 3D mesh from a single input image using marching cubes.
        Uses the same camera parameters as render_video to ensure consistent results.
        
        Args:
            image_path: Path to input image
            output_format: "obj" or "ply" (default: "obj")
            resolution: Grid resolution for marching cubes (default: 128)
            bounds: [[x_min, y_min, z_min], [x_max, y_max, z_max]] bounding box (optional, auto-detected if None)
            isosurface: Isosurface threshold for marching cubes (default: 50.0)
            radius: Camera distance for encoding (default: 1.3, same as render_video)
            focal_val: Focal length for encoding (default: 131.25, same as render_video)
            z_near: Near plane depth (default: 0.8, same as render_video)
            z_far: Far plane depth (default: 1.8, same as render_video)
        
        Returns:
            Path to saved mesh file
        """
        import mcubes
        
        # Load and preprocess image (same as render_video)
        root, ext = os.path.splitext(image_path)
        norm_image_path = f"{root}_normalize{ext}"
        target_path = norm_image_path if os.path.exists(norm_image_path) else image_path
        print(f"Extracting mesh from: {target_path}")

        image = Image.open(target_path).convert("RGB")
        in_sz = 128
        transform = transforms.Compose([
            transforms.Resize((in_sz, in_sz)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        image_tensor = transform(image).unsqueeze(0).unsqueeze(0).to(self.device)

        # Setup camera pose for encoding
        cam_pose = torch.eye(4, device=self.device)
        cam_pose[2, -1] = radius
        focal = torch.tensor(focal_val, dtype=torch.float32, device=self.device)

        # Encode the image into the model
        print("Encoding image...")
        with torch.no_grad():
            self.model.encode(
                image_tensor,
                cam_pose.unsqueeze(0).unsqueeze(0),
                focal
            )
            print(f"Model encoded: {self.model.num_objs} objects, {self.model.num_views_per_obj} views per object")

        # Extract mesh using marching cubes
        print(f"Extracting mesh at resolution {resolution}...")
        print(f"Using camera parameters: radius={radius}, z_near={z_near}, z_far={z_far}, focal={focal_val}")
        reso = [resolution, resolution, resolution]
        
        # Check if model uses viewdirs
        use_viewdirs = getattr(self.model, 'use_viewdirs', False)
        
        # Store provided bounds as fallback (will be used if detection fails)
        fallback_bounds = bounds if bounds is not None else [[-1, -1, -1], [1, 1, 1]]
        
        # Create a wrapper class for marching_cubes that matches expected interface
        class ModelWrapper:
            def __init__(self, model, device, use_viewdirs):
                self.model = model
                self.device = device
                self.use_viewdirs = use_viewdirs
                self.training = False
            
            def eval(self):
                self.training = False
                return self
            
            def train(self):
                self.training = True
                return self
            
            def parameters(self):
                """Expose model parameters for device detection"""
                return self.model.parameters()
            
            def __call__(self, points, coarse=False, viewdirs=None):
                """Query model for density values. Returns output with sigma at index 3."""
                # Ensure points are on the correct device
                if not isinstance(points, torch.Tensor):
                    points = torch.tensor(points, device=self.device)
                elif points.device != self.device:
                    points = points.to(self.device)
                
                # Points shape: (N, 3) -> need (SB=1, B=N, 3)
                if len(points.shape) == 2:
                    points = points.unsqueeze(0)  # (1, N, 3)
                
                with torch.no_grad():
                    if viewdirs is not None:
                        # Ensure viewdirs are on the correct device
                        if not isinstance(viewdirs, torch.Tensor):
                            viewdirs = torch.tensor(viewdirs, device=self.device)
                        elif viewdirs.device != self.device:
                            viewdirs = viewdirs.to(self.device)
                        
                        # Viewdirs shape: (N, 3) -> need (SB=1, B=N, 3)
                        if len(viewdirs.shape) == 2:
                            viewdirs = viewdirs.unsqueeze(0)  # (1, N, 3)
                        outputs = self.model(points, coarse=coarse, viewdirs=viewdirs)
                    else:
                        outputs = self.model(points, coarse=coarse)
                    
                    # Outputs shape: (SB=1, B=N, 4) where last dim is [r, g, b, sigma]
                    # Return shape: (N, 4) for marching_cubes
                    if len(outputs.shape) == 3:
                        outputs = outputs.squeeze(0)  # (N, 4)
                    
                    return outputs
        
        model_wrapper = ModelWrapper(self.model, self.device, use_viewdirs)
        
        # Helper function to query model with viewdirs handling
        def query_model(points_tensor, use_viewdirs_flag):
            """Query model for sigma values, handling viewdirs if needed"""
            if len(points_tensor.shape) == 2:
                points_tensor = points_tensor.unsqueeze(0)  # (1, N, 3)
            
            with torch.no_grad():
                if use_viewdirs_flag:
                    # Create fake viewdirs pointing from points to origin (normalized)
                    points_flat = points_tensor.squeeze(0)  # (N, 3)
                    fake_viewdirs = -points_flat / (torch.norm(points_flat, dim=-1, keepdim=True) + 1e-8)
                    fake_viewdirs = fake_viewdirs.unsqueeze(0)  # (1, N, 3)
                    outputs = self.model(points_tensor, coarse=False, viewdirs=fake_viewdirs)
                else:
                    outputs = self.model(points_tensor, coarse=False)
            
            if len(outputs.shape) == 3:
                outputs = outputs.squeeze(0)  # (N, 4)
            return outputs[:, 3]  # Return only sigma values
        
        # Step 1: Coarse sampling to find object bounds
        print("Locating object bounds with coarse sampling...")
        coarse_res = 32  # Lower resolution for initial search
        
        # Use bounds that match render_video's viewing frustum
        # render_video samples rays from z_near=0.8 to z_far=1.8
        # Camera is at radius=1.3, looking at origin
        # Object should be roughly centered at origin, within the viewing frustum
        # Estimate XY span from field of view: FOV ≈ 2*atan(image_half_size/focal)
        # At average depth (z_near+z_far)/2 = 1.3, span ≈ 1.3 * tan(FOV/2)
        image_half_size = 64  # 128/2
        fov_half = np.arctan(image_half_size / focal_val)
        avg_depth = (z_near + z_far) / 2
        xy_span = avg_depth * np.tan(fov_half) * 1.3  # Add margin
        
        # Depth: object spans roughly from -0.5 to 0.5 (centered at origin)
        # But use the full z_near to z_far range as reference
        depth_span = (z_far - z_near) / 2
        
        # Use bounds centered at origin, matching render_video's viewing frustum
        coarse_bounds = [
            [-xy_span, -xy_span, -depth_span],
            [xy_span, xy_span, depth_span]
        ]
        print(f"Initial coarse bounds (matching render_video frustum): {coarse_bounds}")
        
        # Sample a coarse grid
        x_coarse = torch.linspace(coarse_bounds[0][0], coarse_bounds[1][0], coarse_res, device=self.device)
        y_coarse = torch.linspace(coarse_bounds[0][1], coarse_bounds[1][1], coarse_res, device=self.device)
        z_coarse = torch.linspace(coarse_bounds[0][2], coarse_bounds[1][2], coarse_res, device=self.device)
        
        # Create grid
        xx, yy, zz = torch.meshgrid(x_coarse, y_coarse, z_coarse, indexing='ij')
        grid_points = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)  # (N, 3)
        
        # Query in batches
        batch_size = 10000
        all_sigmas = []
        for i in range(0, len(grid_points), batch_size):
            batch_points = grid_points[i:i+batch_size]
            batch_sigmas = query_model(batch_points, use_viewdirs)
            all_sigmas.append(batch_sigmas)
        
        all_sigmas = torch.cat(all_sigmas)
        sigma_grid = all_sigmas.view(coarse_res, coarse_res, coarse_res)
        
        # Find where density is above a threshold (use a higher percentile to filter noise)
        # Use a higher percentile to focus on the core object and exclude noise
        sigma_threshold = torch.quantile(all_sigmas, 0.88).item()  # Top 15% of densities (more selective)
        object_mask = sigma_grid > sigma_threshold
        
        # Also filter out small disconnected regions (noise)
        # Find the largest connected component
        try:
            from scipy import ndimage
            labeled_mask, num_features = ndimage.label(object_mask.cpu().numpy())
            if num_features > 0:
                # Find the largest component
                component_sizes = [np.sum(labeled_mask == i) for i in range(1, num_features + 1)]
                largest_component = np.argmax(component_sizes) + 1
                object_mask = torch.tensor(labeled_mask == largest_component, device=self.device)
                print(f"Filtered to largest connected component (removed {num_features - 1} noise regions)")
        except ImportError:
            print("scipy not available, skipping connected component filtering")
        
        if object_mask.sum() == 0:
            print("Warning: No high-density region found, using fallback bounds")
            c1, c2 = fallback_bounds[0], fallback_bounds[1]
        else:
            # Find bounding box of object region
            coords = torch.nonzero(object_mask, as_tuple=False).float()
            # Convert grid indices to world coordinates
            for dim in range(3):
                coords[:, dim] = coords[:, dim] / (coarse_res - 1) * (coarse_bounds[1][dim] - coarse_bounds[0][dim]) + coarse_bounds[0][dim]
            
            # Calculate bounding box
            bbox_min = [coords[:, 0].min().item(), coords[:, 1].min().item(), coords[:, 2].min().item()]
            bbox_max = [coords[:, 0].max().item(), coords[:, 1].max().item(), coords[:, 2].max().item()]
            
            # Add relative padding (15% of bounding box size on each side)
            padding_factor = 0.15
            c1 = [
                bbox_min[0] - (bbox_max[0] - bbox_min[0]) * padding_factor,
                bbox_min[1] - (bbox_max[1] - bbox_min[1]) * padding_factor,
                bbox_min[2] - (bbox_max[2] - bbox_min[2]) * padding_factor
            ]
            c2 = [
                bbox_max[0] + (bbox_max[0] - bbox_min[0]) * padding_factor,
                bbox_max[1] + (bbox_max[1] - bbox_min[1]) * padding_factor,
                bbox_max[2] + (bbox_max[2] - bbox_min[2]) * padding_factor
            ]
            
            # Ensure bounds are valid (min < max)
            for i in range(3):
                if c1[i] >= c2[i]:
                    # Fallback to fallback_bounds if invalid
                    c1[i] = fallback_bounds[0][i]
                    c2[i] = fallback_bounds[1][i]
                # Ensure minimum size (at least 0.1 units)
                if c2[i] - c1[i] < 0.1:
                    center = (c1[i] + c2[i]) / 2
                    c1[i] = center - 0.05
                    c2[i] = center + 0.05
            
            print(f"Found object bounds: {c1} to {c2}")
        
        # Step 2: Sample points in the refined bounds to determine isosurface threshold
        print("Sampling sigma values to determine appropriate isosurface threshold...")
        test_points = torch.rand(5000, 3, device=self.device)
        # Scale to bounds
        for i in range(3):
            test_points[:, i] = test_points[:, i] * (c2[i] - c1[i]) + c1[i]
        
        test_sigmas = query_model(test_points, use_viewdirs)
        sigma_min = test_sigmas.min().item()
        sigma_max = test_sigmas.max().item()
        sigma_mean = test_sigmas.mean().item()
        sigma_std = test_sigmas.std().item()
        print(f"Sigma statistics: min={sigma_min:.4f}, max={sigma_max:.4f}, mean={sigma_mean:.4f}, std={sigma_std:.4f}")
        
        # Auto-adjust isosurface if default seems wrong
        sigma_percentile_50 = torch.quantile(test_sigmas, 0.5).item()
        sigma_percentile_75 = torch.quantile(test_sigmas, 0.75).item()
        sigma_percentile_90 = torch.quantile(test_sigmas, 0.90).item()
        print(f"Sigma percentiles: 50%={sigma_percentile_50:.4f}, 75%={sigma_percentile_75:.4f}, 90%={sigma_percentile_90:.4f}")
        
        # Use a higher percentile-based threshold to filter out noise
        # Focus on the core object (75th-85th percentile range) rather than including noise
        if isosurface > sigma_max or isosurface < sigma_min:
            # Use a value in the upper range to focus on the object and exclude noise
            # Weight towards 85th percentile to be more selective
            suggested_threshold = sigma_percentile_75 + (sigma_percentile_90 - sigma_percentile_75) * 0.7
            print(f"Auto-adjusting isosurface threshold from {isosurface} to {suggested_threshold:.4f}")
            isosurface = suggested_threshold
        else:
            # Even if within range, prefer a higher threshold to reduce noise
            # Use the higher of: current threshold or 75th percentile
            if isosurface < sigma_percentile_75:
                print(f"Increasing isosurface threshold from {isosurface:.4f} to {sigma_percentile_75:.4f} to reduce noise")
                isosurface = sigma_percentile_75
        
        # Run marching cubes (sigma is at index 3 in the output)
        print(f"Running marching cubes with isosurface threshold: {isosurface}")
        vertices, triangles = marching_cubes(
            model_wrapper,
            c1=c1,
            c2=c2,
            reso=reso,
            isosurface=isosurface,
            sigma_idx=3,  # Sigma is at index 3 in model output [r, g, b, sigma]
            eval_batch_size=self.render_batch_size,
            coarse=False,
            device=self.device
        )
        
        print(f"Mesh extraction complete: {len(vertices)} vertices, {len(triangles)} triangles")
        
        # Post-process: Remove small disconnected components (noise)
        if len(vertices) > 0 and len(triangles) > 0:
            try:
                # Ensure numpy arrays
                if isinstance(vertices, torch.Tensor):
                    vertices = vertices.cpu().numpy()
                if isinstance(triangles, torch.Tensor):
                    triangles = triangles.cpu().numpy()
                vertices, triangles = self._remove_small_components(vertices, triangles)
                print(f"After noise removal: {len(vertices)} vertices, {len(triangles)} triangles")
            except Exception as e:
                print(f"Warning: Could not remove small components: {e}")
        
        if len(vertices) == 0 or len(triangles) == 0:
            print("Warning: No mesh extracted! Trying with lower isosurface threshold...")
            # Try with progressively lower thresholds
            # Use the sigma statistics we computed earlier
            trial_thresholds = []
            if 'sigma_mean' in locals():
                trial_thresholds = [sigma_mean, sigma_mean * 0.5, sigma_mean * 0.1, max(0.1, sigma_min)]
            else:
                # Fallback: try common thresholds
                trial_thresholds = [10.0, 5.0, 1.0, 0.5, 0.1]
            
            for trial_threshold in trial_thresholds:
                if trial_threshold <= 0:
                    continue
                print(f"Retrying with threshold: {trial_threshold:.4f}")
                try:
                    vertices, triangles = marching_cubes(
                        model_wrapper,
                        c1=c1,
                        c2=c2,
                        reso=reso,
                        isosurface=trial_threshold,
                        sigma_idx=3,
                        eval_batch_size=self.render_batch_size,
                        coarse=False,
                        device=self.device
                    )
                    if len(vertices) > 0 and len(triangles) > 0:
                        print(f"Success with threshold {trial_threshold:.4f}: {len(vertices)} vertices, {len(triangles)} triangles")
                        break
                except Exception as e:
                    print(f"Error with threshold {trial_threshold:.4f}: {e}")
                    continue
            
            if len(vertices) == 0 or len(triangles) == 0:
                raise RuntimeError(f"Failed to extract mesh with any threshold. Last sigma stats: min={sigma_min if 'sigma_min' in locals() else 'N/A'}, max={sigma_max if 'sigma_max' in locals() else 'N/A'}, mean={sigma_mean if 'sigma_mean' in locals() else 'N/A'}")

        # Save mesh
        im_name = os.path.basename(root)
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        if output_format.lower() == "ply":
            # Save as PLY format
            try:
                import mcubes
                mesh_path = os.path.join(output_dir, im_name + "_mesh.ply")
                mcubes.export_obj(vertices, triangles, mesh_path.replace('.ply', '_temp.obj'))
                # Convert OBJ to PLY (simple approach - save as PLY)
                # For proper PLY, we'd need additional library or manual writing
                # Using mcubes export_mesh if available
                if hasattr(mcubes, 'export_mesh'):
                    mcubes.export_mesh(vertices, triangles, mesh_path)
                else:
                    # Fallback: save as OBJ and rename, or write PLY manually
                    # For now, let's write a simple PLY file
                    self._write_ply(vertices, triangles, mesh_path)
                print(f"Mesh saved: {mesh_path}")
                return mesh_path
            except Exception as e:
                print(f"Warning: Could not save as PLY ({e}), saving as OBJ instead")
                output_format = "obj"

        # Save as OBJ format (default)
        mesh_path = os.path.join(output_dir, im_name + "_mesh.obj")
        save_obj(vertices, triangles, mesh_path)
        print(f"Mesh saved: {mesh_path}")
        
        return mesh_path

    def _remove_small_components(self, vertices, triangles, min_faces=100):
        """
        Remove small disconnected mesh components (noise).
        
        Args:
            vertices: (N, 3) numpy array of vertices
            triangles: (M, 3) numpy array of triangle indices
            min_faces: Minimum number of faces to keep a component (default: 100)
        
        Returns:
            Cleaned vertices and triangles
        """
        try:
            from scipy.sparse import csgraph
            import numpy as np
            
            # Build adjacency graph from triangles
            n_vertices = len(vertices)
            
            # Create edge list from triangles
            edges = []
            for tri in triangles:
                edges.append([tri[0], tri[1]])
                edges.append([tri[1], tri[2]])
                edges.append([tri[2], tri[0]])
            
            if len(edges) == 0:
                return vertices, triangles
            
            edges = np.array(edges)
            
            # Build sparse adjacency matrix
            from scipy.sparse import csr_matrix
            adj_matrix = csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), 
                                   shape=(n_vertices, n_vertices))
            adj_matrix = adj_matrix + adj_matrix.T  # Make symmetric
            
            # Find connected components
            n_components, labels = csgraph.connected_components(adj_matrix, directed=False)
            
            # Count faces per component
            component_faces = {}
            for i, tri in enumerate(triangles):
                # Get component label of first vertex (all vertices in triangle should be same component)
                comp_label = labels[tri[0]]
                if comp_label not in component_faces:
                    component_faces[comp_label] = []
                component_faces[comp_label].append(i)
            
            # Find largest component(s) that meet minimum size
            component_sizes = {label: len(faces) for label, faces in component_faces.items()}
            valid_components = [label for label, size in component_sizes.items() 
                               if size >= min_faces]
            
            if len(valid_components) == 0:
                # If no component meets minimum, use largest
                valid_components = [max(component_sizes.items(), key=lambda x: x[1])[0]]
            
            # Filter vertices and triangles
            valid_vertices_mask = np.isin(labels, valid_components)
            valid_vertex_indices = np.where(valid_vertices_mask)[0]
            
            # Create mapping from old to new vertex indices
            vertex_map = np.full(n_vertices, -1, dtype=np.int32)
            for new_idx, old_idx in enumerate(valid_vertex_indices):
                vertex_map[old_idx] = new_idx
            
            # Filter triangles
            valid_triangles = []
            for tri in triangles:
                # Check if all vertices are in valid components
                if all(vertex_map[v] >= 0 for v in tri):
                    valid_triangles.append([vertex_map[v] for v in tri])
            
            if len(valid_triangles) == 0:
                return vertices, triangles
            
            # Return filtered vertices and triangles
            filtered_vertices = vertices[valid_vertex_indices]
            filtered_triangles = np.array(valid_triangles, dtype=triangles.dtype)
            
            removed_faces = len(triangles) - len(filtered_triangles)
            if removed_faces > 0:
                print(f"Removed {removed_faces} faces from {n_components - len(valid_components)} small components")
            
            return filtered_vertices, filtered_triangles
            
        except ImportError:
            print("scipy not available for mesh cleaning, skipping")
            return vertices, triangles
        except Exception as e:
            print(f"Error in mesh cleaning: {e}")
            return vertices, triangles
    
    def _write_ply(self, vertices, triangles, path):
        """Write a simple PLY file"""
        with open(path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {len(triangles)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            
            # Write vertices
            for v in vertices:
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Write faces
            for t in triangles:
                f.write(f"3 {t[0]} {t[1]} {t[2]}\n")
