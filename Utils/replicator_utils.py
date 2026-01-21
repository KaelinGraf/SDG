#Helper functions for scene_building that relate to Omniverse Replicator. 
#This includes setting up/randomising sensors, defining materials, and defining annotators/writers

import isaacsim.zivid as zivid_sim
from isaacsim.zivid.camera import SamplingMode, ZividCamera,spawn_zivid_casing,ZividCameraModelName
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
    def __init__(self,bin_bounds,bin_position,focal_length=0.6):
        """
        Use the zivid IsaacSim api to generate a zivid camera, and create a replicator camera to move with it (to capture annotation data).
        The camera is moved n times during the capturing of a scene (after the phyiscs steps have been completed) to generate N unique datapoints per scene.
        This approach is preferable over spawning N cameras, due to the reduced resource usage. 
        The camera is not destroyed between scenes as to minimise the chance of memory leaks.
        """
        self.bin_bounds = bin_bounds
        self.focal_length = focal_length 
        init_translation =Ztransforms.Transform(np.array([bin_position[0],bin_position[1],self.focal_length+bin_position[2]+bin_bounds[2]]), Ztransforms.Rotation.from_euler(np.array([0,np.pi/2,0])))

        zivid_prim = zivid_sim.camera.spawn_zivid_casing(
            prim_path = "/World/ZividCamera",
            world_pose = init_translation,
            make_rigid_body=False
        )
        self.zivid_camera = zivid_sim.camera.ZividCamera(
            prim_path="/World/ZividCamera",
            model_name = zivid_sim.camera.models.ZividCameraModelName.ZIVID_2_PLUS_MR60,
            sampling_mode=SamplingMode.DOWNSAMPLE4X4,
            yaml_path=SETTINGS_YAML
        )
        # w, h = self.zivid_camera.get_camera_resolution().as_tuple()
        # self.w1 = create_viewport_window(
        #     camera_path=self.zivid_camera.get_camera_sensor_prim_path(),
        #     name="Camera",
        #     width=w,
        #     height=h,
        #     dock_preference=ui.DockPreference.RIGHT,
        # )
        # assert self.w1 is not None
        # self.w1.set_active(True)


        
        #init_translation =Ztransforms.Transform(np.array([0,0,focal_length]), Ztransforms.Rotation.from_euler(np.array([0.0,np.pi/2,0])))
        #self.zivid_camera.set_world_pose(init_translation)
        #zivid_prim = zivid_sim.camera.spawn_zivid_casing(
        #    prim_path = "/World/ZividCamera",
        #    world_pose = init_translation
        #)
        # self.rep_cam = rep.create.camera(
        #     position = [0,0,0],
        #     focus_distance = self.focal_length,
        #     f_stop = 1.8,
        #     name = "rep_cam"
        # )

    def init_cam(self):
        self.zivid_camera.initialize()

    def draw_fov_zivid(self):
        pass
        #draw_fov(self.zivid_camera)
    def cam_trigger(self):
        """
        Trigger both cameras to capture their respective data streams. We will trigger on frame, where each frame will
        be a unique camera position after physics simulations (i.e. multiple frames per scene)
        """
        frame = self.zivid_camera.get_frame()
        frame.save_ply(Path("./frame.ply"))
        #visualise frame
        self.visualize_frame(frame)
    
  

    def visualize_frame(self,frame):
        point_cloud = frame.get_data_xyz()
        flat_xyz = point_cloud.reshape(-1, 3)
        valid_mask = np.isfinite(flat_xyz).all(axis=1)
        valid_points = flat_xyz[valid_mask]
        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(valid_points)
        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window()
        visualizer.add_geometry(o3d_pc)
        visualizer.run()
        visualizer.destroy_window()


    def cam_look_at(self,prim_path,prim_pos = None):
        """
        rotate camera to look at a prim by computing the dot product of the camera's forward vector and the vector from the camera to the prim,
        and then using the arccos of the dot product to compute the angle between the two vectors. 
        @args:
            prim_path: reference path to extract prim pose from 
            prim_pos: optional, if prim_pos is provided, it will be used instead of extracting from prim_path
                        format: [[x,y,z],[qw,qx,qy,qz]]
        """
        if prim_pos is None:
            prim = get_prim_at_path(prim_path)
            prim_pos = get_world_pose(prim_path) 
        print(prim_pos[0],prim_pos[1])
        
        cam_pos = self.zivid_camera.get_camera_sensor_world_pose() 
        
        rot = cam_pos.rot.as_quat()#NDArray: A quaternion in the format [w, x, y, z].

        
        

        

        