#Helper functions for scene_building that relate to Omniverse Replicator. 
#This includes setting up/randomising sensors, defining materials, and defining annotators/writers

import isaacsim.zivid as zivid_sim
from isaacsim.zivid.camera import SamplingMode, ZividCamera,spawn_zivid_casing,ZividCameraModelName,CameraManager
from isaacsim.zivid import transforms as Ztransforms
import omni.replicator.core as rep
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.xforms import get_world_pose
#import zivid
import open3d as o3d
import numpy as np
from omni.kit.viewport.utility import create_viewport_window
from isaacsim.zivid.utilities.fov import draw_fov
import omni.ui as ui
from pathlib import Path

SETTINGS_YAML = "/home/kaelin/BinPicking/Pose_R_CNN/configs/zivid_config_specular.yml"

class RepCam:
    def __init__(self,bin_bounds=[0,0,0,0,0,0],bin_position=[0,0,0,0,0,0],focal_length=0.6,output_dir=None):
        """
        Use the zivid IsaacSim api to generate a zivid camera, and create a replicator camera to move with it (to capture annotation data).
        The camera is moved n times during the capturing of a scene (after the phyiscs steps have been completed) to generate N unique datapoints per scene.
        This approach is preferable over spawning N cameras, due to the reduced resource usage. 
        The camera is not destroyed between scenes as to minimise the chance of memory leaks.
        """
        self.bin_bounds = bin_bounds
        self.focal_length = focal_length 
        init_translation =[bin_position[0],bin_position[1],self.focal_length+bin_position[2]+bin_bounds[2],0,0,0]
        self.output_dir = output_dir
        self.camera_manager = CameraManager(
            env_path = "/Env_",
            output_dir = output_dir,
        )
        
        


    def init_cam(self):
        self.camera_manager.initialize_cameras()
        
    def capture(self):
        self.camera_manager.capture_sequence()

    def view_frames(self):
        self.camera_manager.load_assets_from_disk()
    
    def set_pose(self,trans,rot,quat=False,local=False):
        """
        Set pose of capture rig
        If either trans or rot is None, keep current of that type
        Args:
            Trans (np.array): [x,y,z]
            Rot (np.array): [rx,ry,rz] (or quat if quat=True). 
            Quat (bool): Expect quaternion input format [w,x,y,z]
            Local (bool): if local, translation is relative to parent prim
        """

        transform = self.zivid_camera.get_local_pose()

        if not quat and rot is not None:
            zivid_rot = Ztransforms.Rotation.from_euler(euler=rot,degrees=True)
            transform.rot = zivid_rot
        elif quat and rot is not None:
            zivid_rot = Ztransforms.Rotation.from_quat(quat = rot)
            transform.rot = zivid_rot
            
        if trans is not None:
            transform.t = trans
        
        self.zivid_camera.set_local_pose(transform)

        




        
        

        

        