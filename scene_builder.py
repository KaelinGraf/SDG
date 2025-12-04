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
import carb
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
from isaacsim.core.api.materials import OmniPBR
from isaacsim.sensors.rtx import apply_nonvisual_material
from pxr import UsdGeom, Gf, UsdPhysics,UsdShade,Sdf, PhysxSchema, Vt, Usd
from omni.physx.scripts import utils
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
from Utils.replicator_utils import RepCam
from Utils.material_manager import MaterialManager
# --- FIX START: REGISTER PATHS BEFORE LOADING MATERIAL LIB ---
# In scene_builder.py

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
    
    print(f"[SceneBuilder] Registered MDL Paths: {final_paths}")

# Call this BEFORE enable_extension("omni.kit.material.library")
register_core_mdl_paths()

# 3. NOW enable the extension. It will see the paths during startup.
enable_extension("omni.kit.material.library")

class SceneBuilder:
    def __init__(self,scene_name):
        self.scene_name = scene_name
        self.objects = {}
        self.read_configs()
        self.asset_manager = AssetManager(self.objects_config)
        self.material_manager = MaterialManager()
        self.world_setup()


    
    def world_setup(self):
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
        scene.CreateGravityMagnitudeAttr().Set(4.810)
        self.world.set_simulation_dt(physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)
        self.world.get_physics_context().set_solver_type("TGS")

        

    def material_setup(self):
        """
        TODO: Impliment material generation (high friction metal-realistic physics properties)
            also must have randomisation graph for specular attributes
            Materials will be dynamically bound to primitive.

            Also: impliment bin material here
        """


        pass
    def data_generator_loop(self,iters):
        #main data generation loop
        #randomises certain scene parameters based on config. 
        #clears and repopulates the scene each iteration
        
        self.world.clear()
        self.world.scene.add_default_ground_plane()
        self.rep_cam = RepCam(self.scene_config.get("bin_dimensions",None))
        for i in range(iters):
            print(f"Starting data generation iteration {i+1}/{iters}")
            self.material_manager.reset()
            self.material_manager.create_material(template="plastic_standardized_surface_finish")
            self.material_manager.populate_materials(n=4)
            self.populate_scene()
            self.world.reset()
            for j in range(1000000):
                self.world.step(render=True)
            self.rep_cam.cam_look_at("/World/Bin")
            #self.rep_cam.cam_trigger()
            
            for obj in self.scene_objects:
                prims.delete_prim(prims.get_prim_path(obj))
            prims.delete_prim(prim_path = "/World/Bin")
            for z in range(500):
                self.world.step(render=True)
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
            
            if self.scene_config.get("vary_object_scale",False):
                obj_diag_length *= max_scale #scale the diag length by the max scale factor
            if obj_diag_length > max_diag_length:
                max_diag_length = obj_diag_length
            
            
        max_diag_length = max_diag_length* 1.25 + self.scene_config.get("voxel_jitter", 0.0) * 2.0 #add jitter to voxel size to ensure no overlaps
       
        
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
                scale_factor *= adjust_scale * 1.1
            print(f"Randomized bin dimensions to: {bin_dims}")
        print(f"Scale factor: {scale_factor}")
        start_x = -bin_dims[0] / 2.0
        start_y = -bin_dims[1] / 2.0
        bin_usd = self.scene_config.get("usd_filepath",None)
        if bin_usd is not None:
            print("creating bin prim")
            self.bin_prim = prims.create_prim(
                prim_path="/World/Bin",
                prim_type="Xform",
                position=(0,0,0),
                usd_path=bin_usd,
                semantic_label="bin"
            )
            bin_xform = UsdGeom.Xformable(self.bin_prim)
            bin_xform.ClearXformOpOrder() 
            bin_xform.AddTranslateOp().Set(Gf.Vec3d(start_x,start_y,0))
            bin_xform.AddRotateXYZOp().Set(Gf.Vec3d(0,0,0))
            bin_xform.AddScaleOp().Set(Gf.Vec3d(scale_factor, scale_factor, scale_factor))
            self.generate_box_uvs(self.bin_prim)
            self.material_manager.bind_material(mat_prim_path="/Looks/Plastic_Standardized_Surface_Finish_V15",prim_path="/World/Bin")
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

        voxel_cells = list(range(0, num_cells_x * num_cells_y * num_cells_z))
        random.shuffle(voxel_cells)

        for i, selected_object in enumerate(objects_to_add):
            cell_index = voxel_cells.pop()
            
            # 3. FIX: Pass the corrected corner origin
            [cell_x, cell_y, cell_z] = get_position_from_voxel_index(
                voxel_index=(cell_index % num_cells_x, (cell_index // num_cells_x) % num_cells_y, cell_index // (num_cells_x * num_cells_y)),
                voxel_size=(max_diag_length, max_diag_length, max_diag_length),
                grid_origin=(start_x, start_y, spawn_height_z), 
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
                prim_type="Xform"
            )
            self.generate_box_uvs(object_prim)
            #self.material_manager.create_and_bind(prim_path=object_prim.GetPath())
            self.material_manager.bind_material(prim_path=object_prim.GetPath())
            self.assign_physics_materials(object_prim)

            self.scene_objects.append(object_prim)
    def assign_physics_materials(self, prim,is_static=False):
        """
        Applies Rigid Body physics, Mass, Collisions, and Visual Materials.
        """

        # MESH STUFF
        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            
            if not is_static:
                mesh_api.CreateApproximationAttr("convexHull")
            else:
                mesh_api.CreateApproximationAttr("convexDecomposition") #bin gets decomp
        # COLLISION
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            col_api = UsdPhysics.CollisionAPI.Apply(prim)
            col_api.CreateCollisionEnabledAttr(True)
        if not is_static:
            # RIGID BODY
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rb_api = UsdPhysics.RigidBodyAPI.Apply(prim)
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

    
    # def populate_sensor(self):
    #     """
    #     Use the zivid IsaacSim api to generate a zivid camera, and create a replicator camera to move with it (to capture annotation data).
    #     The camera is moved n times during the capturing of a scene (after the phyiscs steps have been completed) to generate N unique datapoints per scene.
    #     This approach is preferable over spawning N cameras, due to the reduced resource usage. 
    #     The camera is not destroyed between scenes as to minimise the chance of memory leaks.
    #     """
    #     self.zivid_camera = zivid_sim.camera.ZividCamera(
    #         prim_path="/World/ZividCamera",
    #         model_name = zivid_sim.camera.models.ZividCameraModelName.ZIVID_2_PLUS_MR60
    #     )
        
        
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
            print(f"Generated Box UVs for {prim.GetPath()}")
    
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