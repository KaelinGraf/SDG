#
import isaacsim.zivid as zivid_sim
import omni.replicator.core as rep
from isaacsim.zivid import transforms as Ztransforms
import numpy as np
import random
import math
from pxr import Gf

class RepCam:
    def __init__(self, bin_bounds, focal_length=0.6):
        self.bin_bounds = bin_bounds
        self.focal_length = focal_length
        
        # 1. Initialize Zivid
        self.zivid_camera = zivid_sim.camera.ZividCamera(
            prim_path="/World/ZividCamera",
            model_name=zivid_sim.camera.models.ZividCameraModelName.ZIVID_2_PLUS_MR60
        )
        self.zivid_camera.initialize()

        # 2. Initialize Replicator
        self.rep_cam_prim = rep.create.camera(
            position=(0, 0, 0),
            focus_distance=focal_length * 1000,
            f_stop=1.8,
            name="RepCamera"
        )
        
        # 3. Annotators
        self.render_product = rep.create.render_product(self.rep_cam_prim, (1920, 1200))
        self.rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb_annotator.attach([self.render_product])
        self.sem_annotator = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")
        self.sem_annotator.attach([self.render_product])
        
        # Warmup Graph
        rep.orchestrator.step()

    def capture_scene(self, bin_pos=(0,0,0), num_views=1):
        captured_data = []
        poses = self.generate_viewpoints_on_hemisphere(bin_pos, self.focal_length, num_views)

        for pos, rot_quat in poses:
            # 1. Move Cameras (Handles Euler conversion internally)
            self.move_cameras(pos, rot_quat)
            
            # Initialize vars to None to prevent NameError on crash
            rgb_data, sem_data, xyz, rgba = None, None, None, None

            # 2. Trigger Replicator
            try:
                rep.orchestrator.step()
                rgb_data = self.rgb_annotator.get_data()
                sem_data = self.sem_annotator.get_data()
            except Exception as e:
                print(f"[RepCam] Replicator Error: {e}")
                # Don't continue, try to get Zivid anyway if partial data is okay
            
            # 3. Trigger Zivid
            try:
                frame = self.zivid_camera.capture()
                xyz = frame.get_data_xyz()
                rgba = frame.get_data_rgb()
            except Exception as e:
                print(f"[RepCam] Zivid Error: {e}")

            # 4. Store Data
            if rgb_data is not None and xyz is not None:
                data_packet = {
                    "position": pos,
                    "rotation": rot_quat,
                    "zivid_xyz": xyz,
                    "zivid_rgba": rgba,
                    "rep_rgb": rgb_data,
                    "rep_semantic": sem_data
                }
                captured_data.append(data_packet)
                print(f"[RepCam] Captured view at {pos}")

        return captured_data

    def move_cameras(self, position, orientation_quat):
        """
        Moves cameras. Converts Quat to Euler for Replicator.
        """
        # 1. Convert Quaternion (w,x,y,z) to Euler (Deg) for Replicator
        quat_gf = Gf.Quatd(orientation_quat[0], orientation_quat[1], orientation_quat[2], orientation_quat[3])
        rot_gf = Gf.Rotation(quat_gf)
        euler_vec = rot_gf.Decompose(Gf.Vec3d.XAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.ZAxis())
        euler_angles = (euler_vec[0], euler_vec[1], euler_vec[2])

        # 2. Move Replicator (Pass Euler)
        with rep.new_layer():
            rep.modify.pose(
                input_prims=self.rep_cam_prim,
                position=position,
                rotation=euler_angles 
            )
            
        # 3. Move Zivid (Pass Quat)
        pos_np = np.array(position, dtype=np.float64)
        quat_np = np.array(orientation_quat, dtype=np.float64) # w, x, y, z

        new_zivid_pose = Ztransforms.Transform(
            pos_np, 
            Ztransforms.Rotation.from_quat(quat_np)
        )
        self.zivid_camera.set_world_pose(new_zivid_pose)

    def generate_viewpoints_on_hemisphere(self, center, radius, num_points):
        # ... (Same logic as before, just ensuring correct lookAt math) ...
        poses = []
        for _ in range(num_points):
            phi = random.uniform(0, 2 * math.pi)
            theta = random.uniform(0, math.pi / 4)
            
            x = center[0] + radius * math.sin(theta) * math.cos(phi)
            y = center[1] + radius * math.sin(theta) * math.sin(phi)
            z = center[2] + radius * math.cos(theta)
            if z < self.bin_bounds[2] + 0.2: z = self.bin_bounds[2] + 0.2
            
            position = (x, y, z)
            
            # Robust LookAt Math
            forward = np.array(center) - np.array(position)
            forward /= np.linalg.norm(forward)
            world_up = np.array([0, 0, 1])
            right = np.cross(forward, world_up)
            if np.linalg.norm(right) < 0.001: right = np.array([1, 0, 0])
            right /= np.linalg.norm(right)
            new_up = np.cross(right, forward)
            
            mat = np.eye(3)
            mat[:, 0] = right
            mat[:, 1] = new_up
            mat[:, 2] = -forward # Cam looks down -Z
            
            gf_mat = Gf.Matrix3d(
                float(mat[0,0]), float(mat[0,1]), float(mat[0,2]),
                float(mat[1,0]), float(mat[1,1]), float(mat[1,2]),
                float(mat[2,0]), float(mat[2,1]), float(mat[2,2])
            )
            quat = gf_mat.ExtractRotation().GetQuat()
            rot_quat = (quat.GetReal(), quat.GetImaginary()[0], quat.GetImaginary()[1], quat.GetImaginary()[2])
            
            poses.append((position, rot_quat))
        return poses