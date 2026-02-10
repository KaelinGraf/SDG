#NVIDIA Isaac Sim based scene generator for Synthetic Data Generation suitable for a perception pipeline
#Renders RGB-D/Pointclouds of each scene along with annotations like segmentation masks, 6D object poses etc (annotations are selectable)
#Directly puprosed to generate data suitable for bin picking tasks
#Objects are randomly oriented and dropped into a bin of random dimensions from a random height, using physics simulation. The parts can be homogenous
#or heterogeneous. 
#The resulting scene is rendered from random camera viewpoints as to maximize training data output quantity for a given amount of simulation time
#Conditions such as surface texture, lighting, camera intrinsics and noise profiles can be randomized to improve model robustness
#Real depth sensor simulation is used in the rendering pipeline to close the sim2real gap by properly mimicking real world sensor noise characteristics
#Certain domain randomization parameters can be toggled to be static if required
#Certain annotation types have augmentations that can be toggled on/off, such as segmentation mask edge noise to better mimic the performance of 
#a segmentation model (if the data is to be used for training a model downstream of a segmentation model in the perception pipeline)
#This data generation pipeline is therefore designed to be as modular and configurable as possible to suit a variety of use cases and requirements

#Author: Kaelin Graf-Ogilvie (ISCAR Pus)
#Email: kaelin@iscar.co.nz

ZIVID_EXT_PATH = "/home/kaelin/zivid-isaac-sim/source"
NUM_CLONES = 16 #MIN 2
HDRI_PATH = "/home/kaelin/BinPicking/SDG/IS/assets/HDRI/"
OUTPUT_DIR = "Outputs"



from isaacsim import SimulationApp
import warp
simulation_app = SimulationApp({
    "headless": False,
    "--/rtx/hydra/descriptorSetLimit": 65535,
    "--/rtx/material/descriptorSetLimit": 65535,
    "--/rtx/translucency/maxNodes": 65535,
    "--/rtx/post/dlss/enabled": False,

}) 
import carb
from isaacsim.core.utils.extensions import enable_extension
import omni.kit.app
extension_manager = omni.kit.app.get_app().get_extension_manager()

extension_manager.add_path(ZIVID_EXT_PATH)
if not enable_extension("isaacsim.zivid"):
    raise RuntimeError("Failed to enable zivid extension")
import omni.replicator.core as rep
import omni.usd
from omni.isaac.core.utils import prims,xforms
from omni.isaac.core import World
from omni.isaac.core.utils.semantics import add_update_semantics
from omni.isaac.core.utils.stage import open_stage

from omni.isaac.core.materials import PhysicsMaterial
from isaacsim.core.api.materials import OmniPBR
from isaacsim.sensors.rtx import apply_nonvisual_material
from pxr import UsdGeom, Gf, UsdPhysics,UsdShade,Sdf, PhysxSchema, Vt, Usd,UsdLux
from omni.physx.scripts import utils
from isaacsim.core.utils.rotations import euler_angles_to_quat
import isaacsim.core.utils.bounds as bounds_utils
import isaacsim.zivid as zivid_sim
from isaacsim.core.cloner import GridCloner    # import GridCloner interface
import omni.isaac.core.utils.semantics as semantics_utils


import numpy as np
import json
import os
import time
import argparse
import random
import math
import logging
from PyQt6.QtWidgets import QApplication, QWidget

from Utils.mesh_utils import AssetManager,get_bounds
from Utils.mesh_utils import get_position_from_voxel_index,get_voxel_positions_vectorised, get_obb,is_point_within_obb
from Utils.replicator_utils import RepCam
from Utils.material_manager import MaterialManager

from pathlib import Path

dir = script_dir = Path(__file__).resolve().parent



def register_core_mdl_paths():
    tokens = carb.tokens.get_tokens_interface()
    kit_path = tokens.resolve("${kit}")
    
    # 1. Define the necessary paths
    # Core definitions (::nvidia::core)
    path_core = os.path.join(kit_path, "mdl", "core", "mdl")
    # Material definitions (::base, ::state, etc.)
    path_materials = os.path.join(kit_path, "mdl", "materials", "mdl")
    # Your local assets
    path_local = "/home/kaelin/Documents/mdl"
    
    paths_to_add = [path_core, path_materials, path_local]

    # 2. Update Settings Safely
    settings = carb.settings.get_settings()
    key = "/renderer/mdl/searchPaths"
    
    # Get raw value (do not force string)
    current_value = settings.get(key)
    
    final_paths = []
    
    # Handle existing paths whether they are a List or a String
    if current_value:
        if isinstance(current_value, list):
            final_paths = list(current_value)
        elif isinstance(current_value, str):
            final_paths = current_value.split(os.pathsep)
            
    # Add new paths if they aren't already there
    for p in paths_to_add:
        if p not in final_paths:
            final_paths.append(p)
            
    # 3. Write back. 
    # Important: Isaac Sim prefers the List format if it started that way.
    settings.set(key, final_paths)
    
    #print(f"[SceneBuilder] Registered MDL Paths: {final_paths}")

register_core_mdl_paths()
enable_extension("omni.kit.material.library")

class SceneBuilder:
    def __init__(self,scene_name,usd_path=None):
        if usd_path is not None:
            self.world_setup_from_usd(usd_path)
            #self.carb_setup()
            self.scene_name = scene_name
            self.objects = {}
            self.read_configs()
            #initialise asset and material managers after stage is loaded such that the stage pointer held in them is valid
            self.output_dir = os.path.join(os.getcwd(),OUTPUT_DIR)
            if not os.path.exists(self.output_dir):
                os.mkdir(self.output_dir)
            print(f"output to {self.output_dir}")
            print(f"cwd: {os.getcwd()}")
            self.timeline = omni.timeline.get_timeline_interface()
            #self.timeline.pause()

            self.asset_manager = AssetManager(self.objects_config)
            self.material_manager = MaterialManager()
            self.physics_mat = self.create_physics_material(self.stage)
            #self.rep_cam = RepCam(focal_length=self.scene_config["cam_z_dist"])
            self._instantiate_bin()
            self.clone_world()
            self.rep_cam = RepCam(focal_length=self.scene_config["cam_z_dist"],output_dir = self.output_dir)
            self.rep_cam.init_cam()


            self.asset_manager.create_generic_pools(num_bins=NUM_CLONES+1, max_parts_per_bin=50,scene_builder = self)
            #self._build_replicator_graph()
            for _ in range(5):
                simulation_app.update()
            #timeline = omni.timeline.get_timeline_interface()
            # subscription = timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
            # int(omni.timeline.TimelineEventType.PLAY), 
            # self.on_play_callback
            #     )


            start = time.time_ns()
            #rep.utils.send_og_event("randomize_scene")
            self._randomize_scene(10)

            #self.rep_cam.capture()

            end = time.time_ns()
            print(f"Capture time: {(end - start)/1e9} seconds for {NUM_CLONES + 1} scenes")    
            #self.rep_cam.view_frames()
            #for _ in range(100):
                #self.on_play_callback()


        

        else:
            #initialise asset and material managers after stage is loaded such that the stage pointer held in them is valid
            self.scene_name = scene_name
            self.objects = {}
            self.read_configs()
            self.asset_manager = AssetManager(self.objects_config)
            self.material_manager = MaterialManager()
            
            self.world_setup()
            #self.carb_setup()
    
        # self.table_bounds = None
        # if usd_path is not None:
        #     cache = bounds_utils.create_bbox_cache()
        #     self.table_bounds = np.array(bounds_utils.compute_aabb(cache,self.scene_config["tabletop_prim"]))
        #     #apply physics to tabletop
        #     self.assign_physics_materials(self.world.stage.GetPrimAtPath(self.scene_config["tabletop_prim"]),is_static=True)
        
        
     

    
    def world_setup(self):
        """
        Use in the event that a pre-existing USD stage is NOT available. 
        """
        self.world = World()
        #self.world.scene.add_default_ground_plane()
        stage = omni.usd.get_context().get_stage()
        scene = UsdPhysics.Scene.Define(stage,Sdf.Path("/World/PhysicsScene"))
        scene_prim = self.world.stage.GetPrimAtPath("/World/PhysicsScene")  
        if not scene_prim.HasAPI(PhysxSchema.PhysxSceneAPI):
            physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
        else:
            physx_scene_api = PhysxSchema.PhysxSceneAPI(scene_prim)
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.810)
        
        self.world.get_physics_context().set_solver_type("TGS")
        self.world.clear()
        self.world.scene.add_default_ground_plane()
        self.material_manager.create_material(template="wood")
        self.material_manager.bind_material(mat_prim_path="/Looks/Wood_Cork",prim_path="/World/defaultGroundPlane")
        #make ground plane invisible
        #ground_plane = self.world.stage.GetPrimAtPath("/World/defaultGroundPlane")
        #ground_plane.GetAttribute("visibility").Set(False)
        ground_plane_light = self.world.stage.GetPrimAtPath("/World/defaultGroundPlane/SphereLight")
        ground_plane_light.GetAttribute("inputs:intensity").Set(150000.0)
        #translate light up and to the right a bit
        ground_plane_light_xform = UsdGeom.Xformable(ground_plane_light)
        ground_plane_light_xform.ClearXformOpOrder()
        ground_plane_light_xform.AddTranslateOp().Set(Gf.Vec3d(0.0,0.0,2.0))
        #create an ambient light
        # light = UsdLux.DomeLight.Define(self.world.stage, "/World/AmbientLight")
        # light.CreateIntensityAttr(200)
        # light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        
    def world_setup_from_usd(self,usd_path):
        """
        Alternative world setup method, which loads a pre-defined USD scene file.
        """
        open_stage(usd_path)
        self.world = World()
        self.world.reset()
        self.stage = omni.usd.get_context().get_stage()

        scene = UsdPhysics.Scene.Define(self.stage,Sdf.Path("/World/PhysicsScene"))
        scene_prim = self.world.stage.GetPrimAtPath("/World/PhysicsScene")  
        if not scene_prim.HasAPI(PhysxSchema.PhysxSceneAPI):
            physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
        else:
            physx_scene_api = PhysxSchema.PhysxSceneAPI(scene_prim)
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.810)
        
        self.world.get_physics_context().set_solver_type("TGS")
        physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)

        # 1. Contact Buffers (Error: "Contact buffer overflow")
        physx_scene_api.CreateGpuMaxRigidContactCountAttr(10 * 1024 * 1024) 
        
        # 2. Patch Buffers (Error: "Patch buffer overflow")
        physx_scene_api.CreateGpuMaxRigidPatchCountAttr(10 * 1024 * 1024) 
        
        # 3. Temp/Heap Buffers (Good practice to increase these too)
        physx_scene_api.CreateGpuHeapCapacityAttr(64 * 1024 * 1024)
        physx_scene_api.CreateGpuTempBufferCapacityAttr(64 * 1024 * 1024)
        physx_scene_api.CreateGpuFoundLostPairsCapacityAttr(64 * 4096 * 4) # Broadphase pairs
        rep.new_layer()
        
    def clone_world(self):
        base_env_path = "/Env"
        cloner = GridCloner(spacing = 50)
        target_paths = cloner.generate_paths(base_env_path, NUM_CLONES)
        cloner.clone(source_prim_path=base_env_path,prim_paths=target_paths,copy_from_source=True,base_env_path = base_env_path)

        
    def data_generator_loop(self,iters=10):
        #main data generation loop
        #randomises certain scene parameters based on config. 
        #clears and repopulates the scene each iteration
        

        self.rep_cam = RepCam(focal_length=self.scene_config["cam_z_dist"])
        self.rep_cam.init_cam()



        # for i in range(iters):
        #     start = time.time_ns()
        #     print(f"Starting data generation iteration {i+1}/{iters}")
        #     self.material_manager.reset()

        #     self.material_manager.create_material(template="plastic_standardized_surface_finish")
        #     self.material_manager.populate_materials(n=3)
        #     self.populate_scene()
            
        #     cam_trans = [self.bin_pos[0],self.bin_pos[1],self.bin_pos[2]+self.bin_dims[2]+0.6]
        #     cam_rot = np.asarray([-90.0,90.0,90.0])
        #     self.rep_cam.set_pose(trans = cam_trans,rot=cam_rot)

        #     self.world.reset()
        #     timeline = omni.timeline.get_timeline_interface()
            
        #     timeline.stop()
        #     self.world.reset()
        #     for _ in range(10):
        #         self.world.step(render=True)
        #     timeline.play()
        #     for _ in range(10):
        #         self.world.step(render=True)
        #     for j in range(150):
        #         self.world.step(render=True)
        #         #self.rep_cam.draw_fov_zivid()
            
        #     #self.rep_cam.zivid_camera.verify_replicator_attachment()
        #     #self.rep_cam.zivid_camera.verify_gamma_linearity()
        #     self.rep_cam.cam_trigger()
        
        #     for obj in self.scene_objects:
        #         prims.delete_prim(prims.get_prim_path(obj))
        #     prims.delete_prim(prim_path = "/World/Bin")
        #     for _ in range(5):
        #         self.world.step(render=True)
 

        #     print(f"Completed data generation iteration {i+1}/{iters}")
        #     end = time.time_ns()
        #     print(f"Iteration time: {(end - start)/1e9} seconds")
        pass     
    
    def populate_scene(self):
        """
        Populate scene is called within the data_generator_loop method.
        It is responsible for:
        - Randomising scene parameters as per config (bin dimensions, surface texture, lighting, camera params etc)
        - Adding objects to the scene based on config (homogeneous/heterogeneous)
        - Using an object spawning grid relative to the scaled bin dimensions to ensure proper placement within the bin 
            - The max dimension of the largest object + 2*jitter (where jitter is a parameter defining x-y-z randomisation) is used to define the grid cell size
            - N cells are defined in the airspace above the bin, where N is the number of objects to be spawned
            - Each object is then randomly assigned to a cell. This ensures no initial overlaps and proper placement within the bin
            - Within each cell, the object is randomly oriented and offset in each axis by a random amount (up to +- jitter)
            - Each object is then assigned a slightly different specular texture to simulate specular effects with ray tracing/depth sensor simulation
        """
        max_scale = 1.0
        min_scale = 1.0
        self.scene_objects = [] #list to hold selected objects for the scene
        objects_to_add = [] #list to hold object names to be added to the scene
        
        #randomise number of objects if specified
        if self.scene_config.get("rand_num_objects", False):
            self.scene_config["num_objects"] = random.randint(
                self.scene_config["min_num_objects"],
                self.scene_config["max_num_objects"]
            )
            #print(f"Randomized number of objects to: {self.scene_config['num_objects']}")
        
        #randomise scale bounds if specified
        if self.scene_config.get("vary_object_scale",False):
            #randomise scale bounds such that each object can be scaled independently, between min and max scale factors
            min_scale = random.uniform(self.scene_config["min_scale_bounds"][0], self.scene_config["min_scale_bounds"][1])
            max_scale = random.uniform(self.scene_config["max_scale_bounds"][0], self.scene_config["max_scale_bounds"][1]) #max scale will be used to scale largest object bbox for voxel grid
        
        #decide which objects to add based on scene config
        if self.scene_config["object_type"] == "homogeneous":
            selected_object = random.choice(self.object_keys) #choose a random object from the available objects (once per scene)
            for i in range(self.scene_config["num_objects"]):
                objects_to_add.append(selected_object)
        for i in range(self.scene_config["num_objects"]):
            if self.scene_config["object_type"] == "heterogeneous":
                selected_object = random.choice(self.object_keys)# choose a random object for each slot in the scene 
                objects_to_add.append(selected_object)
                
        #based on selected objects, find object with largest bounding box diagonal to define voxel grid size, and bin scaling
        max_diag_length = 0.0
        for obj_name in objects_to_add:
            obj_diag_length = self.asset_manager.asset_registry[obj_name]["diag_length"]
            
            if self.scene_config.get("vary_object_scale",False):
                obj_diag_length *= max_scale #scale the diag length by the max scale factor
            if obj_diag_length > max_diag_length:
                max_diag_length = obj_diag_length
            
            
        max_diag_length = max_diag_length* 1.1 + self.scene_config.get("voxel_jitter", 0.0) * 2.0 #add jitter to voxel size to ensure no overlaps
       
        
        bin_usd = self.scene_config.get("usd_filepath",None)
        if bin_usd is not None:
            #print("creating bin prim")
            self.bin_prim = prims.create_prim(
                prim_path="/World/Bin",
                prim_type="Xform",
                position=(0,0,0),
                usd_path=bin_usd,
                semantic_label="bin"
            )
            
            #find table dimensions
            if self.table_bounds is None:
                self.table_bounds = np.zeros(6)
            bin_pos_x = (self.table_bounds[0] + self.table_bounds[3]) /2
            bin_pos_y = (self.table_bounds[1] + self.table_bounds[4]) /2
            bin_pos_z = self.table_bounds[5]

            bin_bounds = np.array(get_bounds("/World/Bin"))
            bin_dims = bin_bounds[3:6] - bin_bounds[0:3]
            bin_xform = UsdGeom.Xformable(self.bin_prim)
            bin_xform.ClearXformOpOrder() 
            translate = bin_xform.AddTranslateOp()
            translate.Set(Gf.Vec3d((bin_pos_x,bin_pos_y,bin_pos_z)))
            #bin_xform.AddRotateXYZOp().Set(Gf.Vec3d(0,0,90))
                    #randomly scale bin dimensions (ensuring min dimension is at least as large as max_diag_length)
            bin_bounds = np.array(get_bounds("/World/Bin"))
            bin_dims = bin_bounds[3:6] - bin_bounds[0:3]
            self.bin_dims = bin_dims
            self.bin_pos = [bin_pos_x,bin_pos_y,bin_pos_z]
            #print(f"bin dims: {bin_dims}")
            scale_factor = 1
            if self.scene_config.get("vary_bin_scale", False):
                scale_factor = random.uniform(self.scene_config["bin_scale_range"][0], self.scene_config["bin_scale_range"][1])
                bin_dims = bin_dims * scale_factor
                #ensure smallest dimension is larger than diag_length (as this is spawn cell size)
                min_dim = np.min(bin_dims)
                if min_dim < max_diag_length:
                    adjust_scale = max_diag_length / min_dim
                    bin_dims = bin_dims * adjust_scale * 1.1
                    scale_factor *= adjust_scale * 1.1
                #print(f"Randomized bin dimensions to: {bin_dims}")
            #print(f"Scale factor: {scale_factor}")
            bin_xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(scale_factor, scale_factor, scale_factor))
            translate.Set(Gf.Vec3d((bin_pos_x,bin_pos_y,bin_pos_z + bin_dims[2]/2)))
            bin_bounds = np.array(get_bounds("/World/Bin"))
            bin_dims = bin_bounds[3:6] - bin_bounds[0:3]
            self.bin_dims = bin_dims
            #NOTE: We no longer generate box uvs as we bake them into the USD files directly during preprocessing
            #self.generate_box_uvs(self.bin_prim)
            self.material_manager.bind_material(mat_prim_path="/World/Looks/Plastic_Standardized_Surface_Finish_V15",prim_path="/World/Bin/bin/Visuals/FOF_Mesh_Magenta_Box")
            self.assign_physics_materials(self.bin_prim,is_static=True)
        else:
            raise Exception("No bin USD filepath specified")
            
        #use bin scale factor to change number of objects
        #self.scene_config["num_objects"] = int(self.scene_config["num_objects"] * scale_factor)
            
        #create voxel grid, starting from the top of the bin, with cell size = max_diag_length
        
        spawn_height_z = bin_dims[2] # Start spawning at the top rim of the bin

        # Create voxel grid
        num_cells_x = int(bin_dims[0] // max_diag_length)
        num_cells_y = int(bin_dims[1] // max_diag_length)

        # Safety check: If bin is too small for even one object
        if num_cells_x < 1: num_cells_x = 1
        if num_cells_y < 1: num_cells_y = 1

        num_cells_z = math.ceil(self.scene_config["num_objects"] / (num_cells_x * num_cells_y))

        #check total volume does not exceed bin volume by a lot
        total_volume = num_cells_x * num_cells_y * num_cells_z * max_diag_length ** 3
        bin_volume = bin_dims[0] * bin_dims[1] * bin_dims[2]
        if total_volume > bin_volume * 5:
            acceptable_cells = math.ceil(bin_volume / (max_diag_length ** 3)) * 5
            objects_to_add = objects_to_add[:acceptable_cells] #take n first objects 
            num_cells_z = math.ceil(acceptable_cells / (num_cells_x * num_cells_y))
        

        voxel_cells = list(range(0, num_cells_x * num_cells_y * num_cells_z))
        random.shuffle(voxel_cells)

        for i, selected_object in enumerate(objects_to_add):
            cell_index = voxel_cells.pop()
            
            # # 3. FIX: Pass the corrected corner origin
            # [cell_x, cell_y, cell_z] = get_position_from_voxel_index(
            #     voxel_index=(cell_index % num_cells_x, (cell_index // num_cells_x) % num_cells_y, cell_index // (num_cells_x * num_cells_y)),
            #     voxel_size=(max_diag_length, max_diag_length, max_diag_length),
            #     grid_origin=(bin_pos_x,bin_pos_y,bin_pos_z+spawn_height_z),#(bin_pos_x,bin_pos_y, spawn_height_z + bin_pos_z), 
            #     jitter=self.scene_config["voxel_jitter"]
            # )
            idx_x = cell_index % num_cells_x
            idx_y = (cell_index // num_cells_x) % num_cells_y
            idx_z = cell_index // (num_cells_x * num_cells_y)
            [cell_x, cell_y, cell_z] = get_position_from_voxel_index(
                voxel_index=(idx_x, idx_y, idx_z),
                voxel_size=(max_diag_length, max_diag_length, max_diag_length),
                # grid_origin is the center of the bin in X/Y, and the top rim in Z
                grid_origin=(bin_pos_x, bin_pos_y, bin_pos_z + spawn_height_z), 
                # NEW: Pass grid counts so the function knows how far to offset to center the grid
                grid_counts=(num_cells_x, num_cells_y, num_cells_z),
                jitter=self.scene_config["voxel_jitter"]
            )
            #compute scale within scene scaling bounds
            scale_factor_part = random.uniform(min_scale, max_scale) if self.scene_config.get("vary_object_scale",False) else 1.0
            object_prim = prims.create_prim(
                prim_path=f"/World/{selected_object}_{i}",
                position=(cell_x, cell_y, cell_z), #objects are positioned within their respective voxel (with jitter)
                orientation=euler_angles_to_quat([random.uniform(-math.pi, math.pi), random.uniform(-math.pi, math.pi), random.uniform(-math.pi, math.pi)]),
                scale = (scale_factor_part, scale_factor_part, scale_factor_part), #uniformly scale to preserve proportions
                usd_path=self.objects_config[selected_object]["usd_filepath"],
                semantic_label=self.objects_config[selected_object]["class"],
                prim_type="Xform",
                #attributes={"instanceable": True}
            )
            #self.generate_box_uvs(object_prim)
            #self.material_manager.create_and_bind(prim_path=object_prim.GetPath())
            self.material_manager.bind_material(prim_path=object_prim.GetPath())
            self.assign_physics_materials(object_prim)

            self.scene_objects.append(object_prim)
    def assign_physics_materials(self, prim,is_static=False,is_bin=False):
        """
        Applies Rigid Body physics, Mass, Collisions, and Visual Materials.
        """

        # MESH STUFF
    
        if not is_bin and not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            
            if not is_static:
                mesh_api.CreateApproximationAttr("convexHull")
            else:
                mesh_api.CreateApproximationAttr("meshSimplification")
                
        # COLLISION
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            col_api = UsdPhysics.CollisionAPI.Apply(prim)
            col_api.CreateCollisionEnabledAttr(True)
        if not is_static:
            # RIGID BODY
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rb_api = UsdPhysics.RigidBodyAPI.Apply(prim)
            else:
                rb_api = UsdPhysics.RigidBodyApi(prim)
            rb_api.CreateRigidBodyEnabledAttr(True)
            rb_api.CreateKinematicEnabledAttr(False)      
            # MASS
            if not prim.HasAPI(UsdPhysics.MassAPI):
                mass_api = UsdPhysics.MassAPI.Apply(prim)
                mass_api.CreateDensityAttr(7800) #constant steel density for now

            #
            if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                physx_rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                physx_rb_api.CreateEnableCCDAttr(True)
                physx_rb_api.CreateSolverPositionIterationCountAttr(8) # Default is usually 4 or 8
                physx_rb_api.CreateSolverVelocityIterationCountAttr(0)
                physx_rb_api.CreateSleepThresholdAttr(0.005) # Prevent sleeping while still settling
                physx_rb_api.CreateLinearDampingAttr(0.5)  # "Air resistance"
                physx_rb_api.CreateAngularDampingAttr(0.5) # Rotational resistance
                physx_rb_api.CreateMaxLinearVelocityAttr(5.0)
                physx_rb_api.CreateMaxDepenetrationVelocityAttr(1.0)
                
        if is_static:
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rb_api = UsdPhysics.RigidBodyAPI(prim)
                rb_api.CreateRigidBodyEnabledAttr(True)
                rb_api.CreateKinematicEnabledAttr(False)      
    
    def _instantiate_bin(self):
        """
        Picks a bin from the assets (randomly), instiantiates it in the scene (ontop of the table), and applies physics materials
        
        """
        table_prim_path = prims.find_matching_prim_paths("/Env/*table*")[0]
        table_prim = self.stage.GetPrimAtPath(table_prim_path)
        self.assign_physics_materials(table_prim,is_static=True)
        print("Table prim path: ", table_prim_path)
        table_bounds = get_bounds(table_prim_path)
        self.table_bounds = np.array(table_bounds)
        bin_spawn_x = -0.1#(table_bounds[0] + table_bounds[3]) /2
        bin_spawn_y = 0#(table_bounds[1] + table_bounds[4]) /2
        bin_spawn_z = table_bounds[5]
        
        bin_usds = os.listdir(os.path.join(script_dir, "assets/bins/"))
        selected_bin_usd = random.choice(bin_usds)
        print("Selected bin USD: ", selected_bin_usd)
        bin_usd_path = os.path.join(script_dir, "assets/bins/",selected_bin_usd,selected_bin_usd + ".usd")
        print("Full bin USD path: ", bin_usd_path)
        
        bin = prims.create_prim(
            prim_path="/Env/Bin",
            prim_type="Xform",
            translation=(bin_spawn_x,bin_spawn_y,bin_spawn_z),
            orientation= (0.0,0.0,0.0,0.0),#as quaternion
            scale=(1.0,1.0,1.0),
            usd_path=bin_usd_path,
            semantic_label="background"
        )
        
        for child in bin.GetChildren()[0].GetChildren(): #structure is /Env/bin/bin_name/children
            if child.GetName() != "SpawnVolume":
                print(f"applying to {child.GetName()}")
                self.assign_physics_materials(child,is_static=True,is_bin=True)
        mat_prim = self.stage.GetPrimAtPath(self.physics_mat)
        UsdShade.MaterialBindingAPI.Apply(bin).Bind(
            UsdShade.Material(mat_prim),
            materialPurpose = "physics"
        )
    
    def on_play_callback(self,event=None):
        print("Play event detected!")
        time_start = time.time_ns()
                # for _ in range(5):
        simulation_app.update()
        self._randomize_scene(10)
        # for _ in range(5):
        #     simulation_app.update()
        self.rep_cam.capture()
        print(f"time elapsed = {(time_start - time.time_ns()) /1e9}")
    def _randomize_scene(self,iteration):
        """
        Randomzes all scene parameters. 
        Uses replicator where possible for efficiency, however, material attributes and part scattering are performed manually due to 
        inneficiencies/inadequacies in the replicator randomizer nodes for these tasks.
        """
        
        if iteration % 10 ==0:
            #randomize HDRI
            textures = [HDRI_PATH + f for f in os.listdir(HDRI_PATH) if f.endswith('.exr')]
            light_prim = self.stage.GetPrimAtPath("/World/DomeLight")
            light_prim.GetAttribute("inputs:texture:file").Set(random.choice(textures))

        active_paths_resultant = {} #stores active paths (we check bounds after settling physics)
        for bin_index in range(NUM_CLONES+1):
            if bin_index ==0:
                bin_path = "/Env/Bin"
            else:
                bin_path = f"/Env_{bin_index-1}/Bin"
                
            #perform bin scaling and rotation
            bin_pos = xforms.get_local_pose(bin_path)[0] #only get translation
            bin_prim = UsdGeom.Xformable(self.stage.GetPrimAtPath(bin_path))
            bin_prim.ClearXformOpOrder()
            #randomly scale bin dimensions
            bin_bounds = np.array(get_bounds(bin_path))
            bin_dims = bin_bounds[3:6] - bin_bounds[0:3]
            scale_factor = [1.0,1.0,1.0]
            
            for idim, dim in enumerate(bin_dims):
                scale_factor[idim] = random.uniform(self.scene_config["bin_scale_range"][0], self.scene_config["bin_scale_range"][1])
                if idim ==2:
                    scale_factor[2] = min(scale_factor[0],scale_factor[1])
                    continue
                if dim * scale_factor[idim] > (abs(self.table_bounds[3+idim] - self.table_bounds[idim])*0.6):
                    print(f"dimension {idim} max is {abs(self.table_bounds[3+idim] - self.table_bounds[idim])}")
                    print(f"scaling reduced to {((abs(self.table_bounds[3+idim] - self.table_bounds[idim]))/dim) * 0.6}")
                    scale_factor[idim] = ((abs(self.table_bounds[3+idim] - self.table_bounds[idim]))/dim) * 0.6 #add 20% margin
                    if scale_factor[idim] < self.scene_config["bin_scale_range"][0]: scale_factor[idim]=self.scene_config["bin_scale_range"][0]
                    
            translate=bin_prim.AddTranslateOp()
            rotate = bin_prim.AddRotateXYZOp()

            scale = bin_prim.AddScaleOp()
            scale.Set(Gf.Vec3d(scale_factor[0], scale_factor[1], scale_factor[2]))
            #do this before rotation to get axis-aligned bounds
            bin_bounds = np.array(get_bounds(bin_path)) #n
            bin_dims = bin_bounds[3:6] - bin_bounds[0:3]
            rotate.Set(Gf.Vec3d(0.0,0.0,random.uniform(-45.0,45.0)))
            translate.Set(Gf.Vec3d(bin_pos[0],bin_pos[1],bin_pos[2]))


            modes = ["HOMOGENEOUS","HOMO_80_20","CHAOS"]
            mat_mode = random.choice(modes)
            mode = random.choice(modes)
            #perform part randomization and scattering
            randomize_all_materials_direct(self.material_manager.mat_params)

            num_active = self.asset_manager.randomize_bin_contents(bin_index=bin_index,mode=mode,mat_mode=mat_mode,available_materials=self.material_manager.materials_in_scene)
            print(f"Randomized contents of bin at path: {bin_path}")
            bin_pos = xforms.get_world_pose(bin_path)[0] #only get translation
            bin_pos_for_barycentre = xforms.get_local_pose(bin_path)[0]
            bin_bounds = np.array(get_bounds(bin_path))
            bin_dims = bin_bounds[3:6] - bin_bounds[0:3]
            print(f"Bin {bin_index} position: {bin_pos}")
            active_paths = self.asset_manager.bin_pools[bin_index][:num_active] #return first n active paths
            print(f"Active paths: {active_paths}")
            bin_prim = self.stage.GetPrimAtPath(bin_path).GetChildren()[0] #assumes bin prim is first child of /Env/Bin
            volume_prim_path = bin_prim.GetPath().pathString + "/SpawnVolume"
            print(f"Volume prim path: {volume_prim_path}")
            self.world.step()
                    #randomize cameras in dome
            rep.modify.pose_orbit(
                barycentre = bin_pos_for_barycentre,
                distance = random.uniform(bin_dims[2]+0.2,bin_dims[2]+0.8),
                azimuth= random.uniform(0,360),
                elevation = random.uniform(60,90),
                look_at_barycentre = True,
                input_prims = [self.rep_cam.camera_manager.container_paths[bin_index]]
            )
     
            rep.randomizer.scatter_3d(
                volume_prims = [volume_prim_path],
                check_for_collisions = True,
                input_prims = active_paths,
            )
            active_paths_resultant[bin_index] = active_paths

            
        #self.timeline.play()
        simulation_app.update()

        for _ in range(150):
            self.world.step(render=False)
            if _ % 10 == 0:
                self.cull_fallen_parts()
        
        self.update_semantic_labels_for_outliers(active_paths_resultant)
        #self.timeline.pause()
                    







    def update_semantic_labels_for_outliers(self, active_paths_resultant):
        stage = omni.usd.get_context().get_stage()

        for bin_index, active_paths in active_paths_resultant.items():
            if bin_index == 0:
                bin_root = "/Env/Bin"
            else:
                bin_root = f"/Env_{bin_index-1}/Bin"
            
            bin_prim = stage.GetPrimAtPath(bin_root)

            target_prim = bin_prim
            for child in Usd.PrimRange(bin_prim):
                if child.IsA(UsdGeom.Mesh):
                    target_prim = child
                    break
            bin_bounds = get_bounds(bin_root)
            #print(f"found bin {bin_root} with bounds {bin_bounds}")
            for path in active_paths:
                part_prim = stage.GetPrimAtPath(path)
                if not part_prim.IsValid(): continue
                trans_attr = part_prim.GetAttribute("xformOp:translate")
                pos = np.asarray(trans_attr.Get()) # Reads current physics location
                if pos is not None:
                #print(f"found part at {path} with location {pos}")
                    is_contained = is_point_within_obb(pos,bin_root,tolerance=0.01) #10cm of tolerance
                #print(f"part {path} is contained: {is_contained}")
                

                if not is_contained:
                    semantics_utils.add_labels(
                        prim=part_prim,
                        labels=["background"],
                        overwrite=True,
                    )  
            
            # max_diag_length = 0.0
            # for path in active_paths:
            #     prim = self.stage.GetPrimAtPath(path)
            #     child = prim.GetChildren()[0]
            #     obj_name = child.GetName()
            #     try:
            #         obj_diag_length = self.asset_manager.asset_registry[obj_name]["diag_length"]
                
            #     except KeyError:
            #         print(f"Warning: No registry entry for object '{obj_name}'")
            #         continue
            #     if obj_diag_length > max_diag_length:
            #         max_diag_length = obj_diag_length
            # spawn_height_z = 0.2 #start spawning 20cm above bin base
            # # Create voxel grid
            # num_cells_x = int(bin_dims[0] // max_diag_length)
            # num_cells_y = int(bin_dims[1] // max_diag_length)

            # # Safety check: If bin is too small for even one object
            # if num_cells_x < 1: num_cells_x = 1
            # if num_cells_y < 1: num_cells_y = 1

            # num_cells_z = math.ceil(num_active / (num_cells_x * num_cells_y))

            # voxel_cells = list(range(0, num_cells_x * num_cells_y * num_cells_z))
            # random.shuffle(voxel_cells)
            # voxel_positions = get_voxel_positions_vectorised(
            #     voxel_size=(max_diag_length, max_diag_length, max_diag_length),
            #     grid_origin=(bin_pos[0], bin_pos[1], bin_pos[2]), 
            #     grid_counts=(num_cells_x, num_cells_y, num_cells_z),
            #     jitter=self.scene_config["voxel_jitter"]
            # )
            # print(voxel_positions)
            # for i, part in enumerate(active_paths):
            #     cell_index = voxel_cells.pop()
            #     [cell_x, cell_y, cell_z] = voxel_positions[cell_index]
            #     print(f"Placing part {part} at cell index {cell_index} with position {[cell_x, cell_y, cell_z]}")
            #     part_xform = UsdGeom.Xformable(self.stage.GetPrimAtPath(part))
            #     part_xform.ClearXformOpOrder()
            #     translate = part_xform.AddTranslateOp()
            #     translation = Gf.Vec3d(cell_x + bin_pos[0], cell_y + bin_pos[1], cell_z + bin_pos[2])
            #     translate.Set(translation)
            #     part_xform.AddRotateXYZOp().Set(Gf.Vec3d(random.uniform(-180.0, 180.0), random.uniform(-180.0, 180.0), random.uniform(-180.0, 180.0)))
            
                
                
                
    def cull_fallen_parts(self):
        """
        Directly checks the 'xformOp:translate' attribute.
        This is ~10-50x faster than get_world_pose() and reads the raw USD data 
        that PhysX writes to every frame.
        """
        # 1. Iterate active pools
        for bin_idx, pool_paths in self.asset_manager.bin_pools.items():
            for prim_path in pool_paths:
                prim = self.stage.GetPrimAtPath(prim_path)
                if not prim.IsValid(): continue
                
                # 2. VISIBILITY CHECK (Fastest exit)
                # If it's already hidden, skip it.
                vis_attr = prim.GetAttribute("visibility")
                if vis_attr.Get() == "invisible":
                    continue

                # 3. DIRECT TRANSLATE ACCESS (The "Live" check)
                # PhysX updates this specific attribute every frame.
                # We trust that the pool objects are children of a Scope (no parent transform),
                # so local Z == world Z.
                trans_attr = prim.GetAttribute("xformOp:translate")
                val = trans_attr.Get() # Returns Gf.Vec3d
                
                # Check Z height (val[2])
                if val is not None and val[2] < -0.2: # -20cm threshold
                    print(f"[Cull] Disabling part {prim_path} at Z={val[2]:.2f}")
                    
                    # A. Disable Physics (Stop Solver)
                    rb = UsdPhysics.RigidBodyAPI(prim)
                    if rb: rb.GetRigidBodyEnabledAttr().Set(False)
                    
                    # B. Hide (Stop Rendering)
                    vis_attr.Set("invisible")
                    
                    # C. Teleport to Safety (Stop Bounds Checks)
                    trans_attr.Set(Gf.Vec3d(0, -10000, 0))
        


    def create_physics_material(self, stage, name="SmoothPlastic"):
        # 1. Define the path
        mat_path = f"/World/Physics_Materials/{name}"
        
        # 2. Create the Material Prim
        if not stage.GetPrimAtPath(mat_path):
            UsdShade.Material.Define(stage, mat_path)
        
        mat_prim = stage.GetPrimAtPath(mat_path)
        
        # 3. Apply Physics Material API
        phys_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
        
        # 4. Set Properties (Tune these to stop sticking!)
        # Static Friction: Resistance to starting movement (0.0 = Ice, 1.0 = Rubber)
        phys_mat.CreateStaticFrictionAttr(0.2) 
        
        # Dynamic Friction: Resistance while sliding
        phys_mat.CreateDynamicFrictionAttr(0.15) 
        
        # Restitution: Bounciness (0.0 = No bounce, 1.0 = Superball)
        phys_mat.CreateRestitutionAttr(0.1) 
        
        return mat_path
    def _build_replicator_graph(self):
        """
        Assembles Rep Randomizer Graph 
        """
    
        MAX_CAPACITY = 50  # Max number of objects in the bin
        all_parts = [p.GetPath().pathString for p in self.stage.GetPrimAtPath("/World/Prim_Library").GetChildren()]
        all_mats = self.material_manager.materials_in_scene
        # pool = rep.randomizer.instantiate(
        #         paths = all_parts,
        #         size = MAX_CAPACITY
        #     )
        
        bin_groups = {}
        spawn_vols = rep.get.prims(path_pattern="/Env_*/Bin/*/SpawnVolume")
        HDRI_PATH = "/home/kaelin/BinPicking/SDG/IS/assets/HDRI/"

        textures = [HDRI_PATH + f for f in os.listdir(HDRI_PATH) if f.endswith('.exr')]

        
        with rep.trigger.on_custom_event(event_name="randomize_scene"):
            pass
                

            
            # rep.randomizer.scatter_3d(
            #     volume_prims = spawn_vols,
            #     check_for_collisions = True,
            #     input_prims = pool,
            # )
            
            #material randomization
            
            # if self.material_manager:
            #     rep.randomizer.randomize_all_materials_direct(
            #         self.material_manager.mat_params
            #     )
                    
       
        
        
    def _create_material_randomizer(self):
        """
        Creates a Replicator material randomizer node to randomize materials in the scene.
        """
        mat_params = self.material_manager.mat_params
        
        

            
    def _assign_material_rep(self):
        """
        Assigns a random material to each spawned prim, depending on some rules:
        - 70% chance of homogeneous material assignment (all objects get same material)
        - 30% chance of heterogeneous material assignment (each object gets different material)
        Also assigns 
        """
        
        stage = omni.usd.get_context().get_stage()
        mode = "HOMO" if random.random()<0.7 else "HETERO" #70% chance of homogeneous material assignment
        
            

        
        
    def read_configs(self):
        #read scene config file
        with open("./Config/scene_config.json", 'r') as f:
            self.scene_config = json.load(f)[self.scene_name]
        #print("Loaded scene config: ", self.scene_config)
        with open("./Config/objects.json", 'r') as f:
            self.objects_config = json.load(f)
        #print("Loaded objects config: ", self.objects_config)
        self.object_keys = list(self.objects_config.keys()) #list to hold object names, for easy random selection
        #print(len(self.objects_config.keys()), " objects available for scene building")
        
    #def _create_bin_

    def generate_box_uvs(self, root_prim, scale=10.0):
        """
        Applies Box Mapping (Tri-planar) UVs to all meshes under root_prim.
        Uses faceVarying interpolation to prevent stretching on sides.
        """
        # Create iterator to find all meshes
        range_iterator = Usd.PrimRange(root_prim)
        
        for prim in range_iterator:
            if not prim.IsA(UsdGeom.Mesh):
                continue
                
            mesh = UsdGeom.Mesh(prim)
            
            # Get mesh data
            points = mesh.GetPointsAttr().Get()
            face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
            face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
            
            if not points or not face_vertex_indices:
                continue

            pv_api = UsdGeom.PrimvarsAPI(prim)
            # Optional: Skip if UVs exist
            # if pv_api.HasPrimvar("st"): continue

            uvs = []
            
            # Iterate over faces to calculate normals and project UVs
            # faceVarying means we generate 1 UV per vertex-index in the face list
            idx_pointer = 0
            for count in face_vertex_counts:
                # Get indices for this face
                face_indices = face_vertex_indices[idx_pointer : idx_pointer + count]
                idx_pointer += count
                
                # Calculate rough Face Normal using first 3 points of the polygon
                # (Assumes planar faces, which is standard)
                p0 = Gf.Vec3f(points[face_indices[0]])
                p1 = Gf.Vec3f(points[face_indices[1]])
                p2 = Gf.Vec3f(points[face_indices[2]])
                
                # Normal = Cross Product of two edges
                v1 = p1 - p0
                v2 = p2 - p0
                normal = Gf.Cross(v1, v2).GetNormalized()
                
                # Determine Dominant Axis for Box Mapping
                # X-projection
                if abs(normal[0]) >= abs(normal[1]) and abs(normal[0]) >= abs(normal[2]):
                    u_idx, v_idx = 1, 2 # Project onto YZ plane
                # Y-projection
                elif abs(normal[1]) >= abs(normal[0]) and abs(normal[1]) >= abs(normal[2]):
                    u_idx, v_idx = 0, 2 # Project onto XZ plane
                # Z-projection
                else:
                    u_idx, v_idx = 0, 1 # Project onto XY plane

                # Generate UVs for this face
                for v_idx_in_face in face_indices:
                    p = points[v_idx_in_face]
                    
                    # Apply Scale
                    u = p[u_idx] * scale
                    v = p[v_idx] * scale
                    
                    uvs.append(Gf.Vec2f(u, v))

            # Apply the Primvar with faceVarying interpolation
            # This allows 1 vertex to have different UVs for different faces (sharp edges)
            pv = pv_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
            pv.Set(uvs)
            #print(f"Generated Box UVs for {prim.GetPath()}")
    
def main():
    parser = argparse.ArgumentParser(description="Isaac Sim Bin Picking Data Generator Scene Builder")
    parser.add_argument('--scene_name', type=str, default="default_scene", help='Name of the scene configuration to use from scene_config.json')
    parser.add_argument('--iters', type=int, default=10, help='Number of data generation iterations to run')
    args = parser.parse_args()
    
    scene_builder = SceneBuilder(args.scene_name,usd_path="/home/kaelin/BinPicking/SDG/IS/assets/single_stage.usd") #initialize scene builder with specified scene config, does not start data generation
    #scene_builder = SceneBuilder(args.scene_name)#args.scene_name,usd_path="/home/kaelin/Desktop/custom_usds/warehouse_ur12e.usd") #initialize scene builder with specified scene config, does not start data generation

    print(f"Scene Builder initialised for scene: {args.scene_name}. Starting data generation loop for {args.iters} iterations.")
    #scene_builder.world.reset()
    #scene_builder.data_generator_loop(iters=20)
    print("Holding simulation open. Press Ctrl+C in terminal to stop.")
    #scene_builder.data_generator_loop(args.iters) #start data generation loop
    # # Create a viewer loop so you can see the bin
    while simulation_app.is_running():
        # This triggers the renderer and physics step
        simulation_app.update() 
        
    #     # Optional: If you want to verify the bin exists, check it here
    #     # if scene_builder.world.scene.get_object("/World/Bin"):
    #     #    pass
    
    
    
@rep.randomizer.register
def randomize_bin_state(bin_groups_map, all_prototypes, all_materials, max_capacity):
    """
    Manages a fixed pool of objects to simulate a variable count bin.
    """
    stage = omni.usd.get_context().get_stage()
    
    # --- 1. GLOBAL FRAME DECISIONS ---
    # Decide rules for this epoch (e.g., Homogenous vs Heterogenous)
    part_mode = "HOMO" if random.random() < 0.5 else "HETERO"
    
    # Material Logic (40% Uniform, 40% Distractor, 20% Chaos)
    r_mat = random.random()
    if r_mat < 0.4: mat_mode = "UNIFORM"
    elif r_mat < 0.8: mat_mode = "DISTRACTOR"
    else: mat_mode = "CHAOS"

    # --- 2. PER-BIN EXECUTION ---
    for bin_name, pattern in bin_groups_map.items():
        
        # A. Resolve the Pool
        # pattern is like "/Replicator/Pool_Bin_0/Ref_*"
        parent_path = pattern.split("/Ref_")[0]
        parent_prim = stage.GetPrimAtPath(parent_path)
        if not parent_prim.IsValid(): continue
        
        pool_prims = [c for c in parent_prim.GetChildren()]
        
        # B. Decide "N" for this bin (The Variable Count)
        wanted_count = random.randint(5, max_capacity)
        
        active_prims = pool_prims[:wanted_count]
        inactive_prims = pool_prims[wanted_count:]
        
        # --- 3. CONFIGURE ACTIVE OBJECTS ---
        
        # Pre-select shared assets if modes require it
        homo_proto = random.choice(all_prototypes) if part_mode == "HOMO" else None
        uniform_mat = random.choice(all_materials) if mat_mode == "UNIFORM" else None
        distractor_main_mat = random.choice(all_materials) if mat_mode == "DISTRACTOR" else None

        for i, prim in enumerate(active_prims):
            # A. Enable Physics & Vis
            _set_physics_state(prim, True)
            
            # B. Set Part Type (Reference Swap)
            refs = prim.GetReferences()
            refs.ClearReferences()
            if part_mode == "HOMO":
                refs.AddReference(homo_proto)
            else:
                refs.AddReference(random.choice(all_prototypes))

            # C. Set Material
            if mat_mode == "UNIFORM":
                _bind_mat(stage, prim, uniform_mat)
            elif mat_mode == "CHAOS":
                _bind_mat(stage, prim, random.choice(all_materials))
            elif mat_mode == "DISTRACTOR":
                # 80% Main, 20% Random
                tgt = distractor_main_mat if random.random() < 0.8 else random.choice(all_materials)
                _bind_mat(stage, prim, tgt)

        # --- 4. HIDE INACTIVE OBJECTS (Teleport to Void) ---
        for prim in inactive_prims:
            _set_physics_state(prim, False)
            # Teleport far away so they don't interact
            xform = UsdGeom.Xformable(prim)
            xform.AddTranslateOp().Set(Gf.Vec3d(0, -10000, 0))

    return True

# --- HELPER FUNCTIONS ---
def _set_physics_state(prim, enabled):
    """Toggles RigidBody and Visibility safely"""
    # 1. Visibility
    imageable = UsdGeom.Imageable(prim)
    if enabled: imageable.MakeVisible()
    else: imageable.MakeInvisible()
    
    # 2. Physics (RigidBodyAPI)
    rb = UsdPhysics.RigidBodyAPI(prim)
    if not rb: rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb.GetRigidBodyEnabledAttr().Set(enabled)

def _bind_mat(stage, prim, mat_path):
    mat_prim = stage.GetPrimAtPath(mat_path)
    if mat_prim:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material(mat_prim))   
        

def randomize_all_materials_direct(mat_params_dict):
    """
    Randomizes all materials in one go using direct USD API.
    100x faster than creating individual Replicator nodes.
    
    Args:
        mat_params_dict (dict): The dictionary from self.material_manager.mat_params
    """
    stage = omni.usd.get_context().get_stage()
    
    for mat_path, params in mat_params_dict.items():
        # 1. Get Prim directly (Instant lookup, no searching)
        prim = stage.GetPrimAtPath(mat_path)
        if not prim.IsValid():
            continue
            
        shader_prim = stage.GetPrimAtPath(f"{mat_path}/Shader")
        if not shader_prim.IsValid():
            continue
            
        shader = UsdShade.Shader(shader_prim)
        
        for param_name, bounds in params.items():
            if len(bounds) != 2 or not isinstance(bounds, list):
                continue
            #print(f"Randomizing {mat_path} param {param_name} within bounds {bounds}")
            
            # Helper to find the input object
            inp = shader.GetInput(param_name)
            if not inp:
                inp = shader.GetInput(f"inputs:{param_name}")
            
            if inp:
                # Randomize
                val = random.uniform(bounds[0], bounds[1])
                inp.Set(val)
                

if __name__ == "__main__":
    main()
    simulation_app.close()