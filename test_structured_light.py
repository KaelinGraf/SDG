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
    def __init__(self):
        
        self.world_setup()


    
    def world_setup(self):
        self.world = World()
        #self.world.scene.add_default_ground_plane()
        stage = omni.usd.get_context().get_stage()
        create_prim(
        prim_path="/World/WhiteWall",
        prim_type="Cube",
        position=(0.0, 0.0, 0.0),
        scale=(2.0, 2.0, 0.1), # 2m wide wall
        attributes={"displayColor": [(1.0, 1.0, 1.0)]}
        )
        
        # Add ambient light to see the camera body
        UsdLux.DomeLight.Define(simulation_app.stage, "/World/AmbientLight").CreateIntensityAttr(100)

        # 3. Create Zivid Camera
        # Positioned 1m away on X-axis, facing the wall (assuming Z-up world)
        camera_prim_path = "/World/ZividCamera"
        zivid = ZividCamera(
            model_name=ZividCameraModelName.ZIVID_2_PLUS_M70,
            prim_path=camera_prim_path
        )
        
        # Set Pose: 1m away, rotated 180 deg to look at origin
        zivid.set_world_pose(
            pose=Gf.Vec3d(1.0, 0, 0),
            orientation=euler_angles_to_quat([0, 0, 180])
        )
        
        # Initialize sensors (Required for render products)
        zivid.initialize()