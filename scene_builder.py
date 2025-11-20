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

from isaacsim import SimulationApp
import warp
simulation_app = SimulationApp({
    "headless": False,
}) 
from isaacsim.core.utils.extensions import enable_extension
import omni.kit.app
extension_manager = omni.kit.app.get_app().get_extension_manager()
extension_manager.add_path(ZIVID_EXT_PATH)
if not enable_extension("isaacsim.zivid"):
    raise RuntimeError("Failed to enable zivid extension")
import omni.replicator.core as rep
import omni.usd
from omni.isaac.core.utils import prims
from omni.isaac.core import World
from omni.isaac.core.utils.semantics import add_update_semantics
from omni.isaac.core.materials import PhysicsMaterial
from pxr import UsdGeom, Gf, UsdPhysics,UsdShade,Sdf
from isaacsim.core.utils.rotations import euler_angles_to_quat
import isaacsim.core.utils.bounds as bounds_utils
import isaacsim.zivid as zivid_sim
import numpy as np
import json
import os
import time
import argparse
import random
import math
import logging
from Utils.mesh_utils import AssetManager
from Utils.mesh_utils import get_position_from_voxel_index


class SceneBuilder:
    def __init__(self,scene_name):
        self.scene_name = scene_name
        self.objects = {}
        self.read_configs()
        self.asset_manager = AssetManager(self.objects_config)
        self.world_setup()

    
    def world_setup(self):
        self.world = World()
        self.world.scene.add_default_ground_plane()
        
        
    def data_generator_loop(self,iters):
        #main data generation loop
        #randomises certain scene parameters based on config. 
        #clears and repopulates the scene each iteration
        for i in range(iters):
            print(f"Starting data generation iteration {i+1}/{iters}")
            self.populate_scene()
            #do rendering stuff here
            while True:
                simulation_app.update() 

            

            
            for obj in self.scene_objects:
                prims.delete_prim(prims.get_prim_path(obj))
            print(f"Completed data generation iteration {i+1}/{iters}")
            
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
            print(f"Randomized number of objects to: {self.scene_config['num_objects']}")
        
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
            print(f"obj diagonal length: {obj_diag_length}")
            if self.scene_config.get("vary_object_scale",False):
                obj_diag_length *= max_scale #scale the diag length by the max scale factor
            if obj_diag_length > max_diag_length:
                max_diag_length = obj_diag_length
            
            print(f"max_diag_length: {max_diag_length}")
        max_diag_length = max_diag_length* 1.1 + self.scene_config.get("voxel_jitter", 0.0) * 2.0 #add jitter to voxel size to ensure no overlaps
        print(f"max diagonal length: {max_diag_length}")
        
        #randomly scale bin dimensions (ensuring min dimension is at least as large as max_diag_length)
        bin_dims = np.asarray(self.scene_config["bin_dimensions"])
        scale_factor = 1
        if self.scene_config.get("vary_bin_scale", False):
            scale_factor = random.uniform(self.scene_config["bin_scale_range"][0], self.scene_config["bin_scale_range"][1])
            bin_dims = bin_dims * scale_factor
            #ensure smallest dimension is larger than diag_length (as this is spawn cell size)
            min_dim = np.min(bin_dims)
            if min_dim < max_diag_length:
                adjust_scale = max_diag_length / min_dim
                bin_dims = bin_dims * adjust_scale * 1.1
            print(f"Randomized bin dimensions to: {bin_dims}")
        
        bin_usd = self.scene_config.get("usd_filepath",None)
        if bin_usd is not None:
            print("creating bin prim")
            self.bin_prim = prims.create_prim(
                prim_path="/World/Bin",
                prim_type="Xform",
                position=(0,0,0),
                scale=(scale_factor,scale_factor,scale_factor),
                usd_path=bin_usd,
            )
            # Apply Rigid Body API
            rb_api = UsdPhysics.RigidBodyAPI.Apply(self.bin_prim)
            rb_api.CreateRigidBodyEnabledAttr(True)
            rb_api.CreateKinematicEnabledAttr(True) 

            # Apply Collision API
            col_api = UsdPhysics.CollisionAPI.Apply(self.bin_prim)
            col_api.CreateCollisionEnabledAttr(True)
            
            # Optional: Apply Mesh Collision (for accurate concave bin shapes)
            # If objects fall through the walls, enable this:
            mesh_api = UsdPhysics.MeshCollisionAPI.Apply(self.bin_prim)
            mesh_api.CreateApproximationAttr("convexDecomposition")
        else:
            raise Exception
            
            
        #create voxel grid, starting from the top of the bin, with cell size = max_diag_length
        num_cells_x = int(bin_dims[0] // max_diag_length)
        num_cells_y = int(bin_dims[1] // max_diag_length)
        num_cells_z = math.ceil(self.scene_config["num_objects"] / (num_cells_x * num_cells_y)) #number of vertical layers needed (round up)
        
        #create a list of all possible voxel cell positions, each object will be randomly assigned to one 
        voxel_cells = list(range(0, num_cells_x * num_cells_y * num_cells_z))
        random.shuffle(voxel_cells) #randomize order
        
        for i, selected_object in enumerate(objects_to_add):
            #randomly select an unoccupied voxel cell
            cell_index = voxel_cells.pop()
            #compute voxel cell position
            [cell_x, cell_y, cell_z] = get_position_from_voxel_index(
                voxel_index=(cell_index % num_cells_x, (cell_index // num_cells_x) % num_cells_y, cell_index // (num_cells_x * num_cells_y)),
                voxel_size=(max_diag_length, max_diag_length, max_diag_length),
                grid_origin=(0, 0, bin_dims[2]),  # Assuming the origin is at (0,0,0), adjust as needed
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
            )
            self.scene_objects.append(object_prim)
    def assign_physics_materials(self):
        #assign material properties to bin and objects. Objects get slightly randomized specular properties to simulate specular effects
        pass
    
    def populate_sensor(self):
        """
        Use the zivid IsaacSim api to generate a zivid camera, and create a replicator camera to move with it (to capture annotation data).
        The camera is moved n times during the capturing of a scene (after the phyiscs steps have been completed) to generate N unique datapoints per scene.
        This approach is preferable over spawning N cameras, due to the reduced resource usage. 
        The camera is not destroyed between scenes as to minimise the chance of memory leaks.
        """
        self.zivid_camera = zivid_sim.ZividCamera(
            prim_path="/world/ZividCamera",
            model_name = zivid_sim.ZividCameraModelName.ZIVID_2_PLUS_MR60
        )
        
        
    def read_configs(self):
        #read scene config file
        with open("./Config/scene_config.json", 'r') as f:
            self.scene_config = json.load(f)[self.scene_name]
        print("Loaded scene config: ", self.scene_config)
        with open("./Config/objects.json", 'r') as f:
            self.objects_config = json.load(f)
        #print("Loaded objects config: ", self.objects_config)
        self.object_keys = list(self.objects_config.keys()) #list to hold object names, for easy random selection
        print(len(self.objects_config.keys()), " objects available for scene building")
        
        

    
def main():
    parser = argparse.ArgumentParser(description="Isaac Sim Bin Picking Data Generator Scene Builder")
    parser.add_argument('--scene_name', type=str, default="default_scene", help='Name of the scene configuration to use from scene_config.json')
    parser.add_argument('--iters', type=int, default=10, help='Number of data generation iterations to run')
    args = parser.parse_args()
    
    scene_builder = SceneBuilder(args.scene_name) #initialize scene builder with specified scene config, does not start data generation
    
    print(f"Scene Builder initialised for scene: {args.scene_name}. Starting data generation loop for {args.iters} iterations.")
    scene_builder.world.reset()

    print("Holding simulation open. Press Ctrl+C in terminal to stop.")
    
    # # Create a viewer loop so you can see the bin
    # while simulation_app.is_running():
    #     # This triggers the renderer and physics step
    #     simulation_app.update() 
        
    #     # Optional: If you want to verify the bin exists, check it here
    #     # if scene_builder.world.scene.get_object("/World/Bin"):
    #     #    pass
    scene_builder.data_generator_loop(args.iters) #start data generation loop
    
    
    
if __name__ == "__main__":
    main()
    simulation_app.close()