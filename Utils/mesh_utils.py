#provides convenience functions for mesh processing and manipulation

import numpy as np
import random
import os 
import json
from pathlib import Path
from pxr import UsdGeom,UsdPhysics,Gf,UsdShade
#from scene_builder import SceneBuilder
import omni.usd
import isaacsim.core.utils.bounds as bounds_utils
import isaacsim.core.utils.prims as prim_utils
import omni.isaac.core.utils.semantics as semantics_utils
class AssetManager:
    """
    The asset manager provides utility functions for working with USD meshes.
    More importantly, it provides a dictionary of object aligned bounding boxes for each possible scene object supported in the scene builder.
    This is useful for the formation of spawning voxels with randomised object scale without the need to repeatedly extract mesh bounds from USD files.
    """
    def __init__(self,objects_config):
        self.objects_config = objects_config
        self.asset_registry = {} #dictionary to hold object bounding boxes by name
        self._get_all_object_bounds()
        
        
    
    
    def _get_all_object_bounds(self):
        prims_path = "/World/Prim_Library"
        cache = bounds_utils.create_bbox_cache()

        for obj_name, obj_info in self.objects_config["parts"].items():
            usd_filepath = obj_info.get("usd_filepath", None)
            prim_path = f"{prims_path}/{self.objects_config['parts'][obj_name]['name']}"
            name = self.objects_config['parts'][obj_name]['name']
            if usd_filepath:
                prim = prim_utils.create_prim(
                    prim_path=prim_path,
                    usd_path=usd_filepath,
                    translation=[1000, 0, 0],
                    semantic_label="part",
                    attributes={
                        #"instanceable": True
                    }
                )
                #prim.SetInstanceable(True)
                                              
                bounds = np.array(bounds_utils.compute_aabb(cache,prim_path))
                #compute the diagonal (eg 0,0,0) to (x,y,z)
                diag_vector = bounds[3:] - bounds[0:3]
                #print(f"diagonal vector: {diag_vector}")
                diag_length = np.linalg.norm(diag_vector) #this is the diameter of a bounding sphere around the object (max distance across the object)
                #store in asset registry
                #print(bounds)
                #print(f"diagonal length: {diag_length}")
                self.asset_registry[name] = {
                    "bounds": bounds,
                    "diag_length": diag_length
                }
                print(f"Registered asset: {name} with bounds: {bounds} and diagonal length: {diag_length}")
                prim_utils.set_prim_visibility(prim,False)
                #delete temp prim
                #prim_utils.delete_prim(prim_path)
                
                
    def create_generic_pools(self, num_bins, max_parts_per_bin, scene_builder):
        """
        Creates 'max_parts_per_bin' generic Xforms for each bin.
        @args:
            num_bins (int): number of bins to create pools for
            max_parts_per_bin (int): maximum number of parts per bin
        """
        stage = omni.usd.get_context().get_stage()
        self.bin_pools = {} # Map bin_index -> list of prim paths

        for bin_idx in range(num_bins):
            pool_path = f"/World/Pools/Bin_{bin_idx}"
            stage.DefinePrim(pool_path, "Scope")
            
            self.bin_pools[bin_idx] = []
            
            for i in range(max_parts_per_bin):
                prim_path = f"{pool_path}/Part_{i}"
                
                # Create a Generic Xform
                prim = stage.DefinePrim(prim_path, "Xform")
                prim.SetInstanceable(False)

                # Add Physics APIs (Rigid Body, Colliders) NOW so they persist
                # (We will enable/disable them later)
                scene_builder.assign_physics_materials(prim)
                # 2. Apply Mass (Parent defines the weight)
                mass_api = UsdPhysics.MassAPI.Apply(prim)
                mass_api.CreateMassAttr(0.1) # Default placeholder mass
                mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
                
                #create semantic label "part"
                semantics_utils.add_labels(
                    prim=prim,
                    labels=["part"],
                    overwrite=True,
                )
                
                # Make Invisible/Disabled by default
                UsdGeom.Imageable(prim).MakeInvisible()
                UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(False)
                mat_prim = scene_builder.stage.GetPrimAtPath(scene_builder.physics_mat)
                phys_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
        
                phys_mat.CreateStaticFrictionAttr(0.2) 

                phys_mat.CreateDynamicFrictionAttr(0.15) 
                
                phys_mat.CreateRestitutionAttr(0.1) 
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    UsdShade.Material(mat_prim),
                    materialPurpose = "physics"
                )
                
                self.bin_pools[bin_idx].append(prim_path)
                
    def randomize_bin_contents(self, bin_index,available_materials, mode="HOMO_80_20",mat_mode="HOMO_80_20"):
        """
        Configures the generic pool for a specific bin to match the desired distribution.
        """
        stage = omni.usd.get_context().get_stage()
        pool_paths = self.bin_pools[bin_index]
        
        # 1. Decide Object Counts
        num_active = random.randint(30, 50) # How many parts in this bin?
        
        # Get all available object USD paths from your config
        all_usd_paths = [v['usd_filepath'] for v in self.objects_config['parts'].values()]
        
        # 2. Select Objects based on Mode
        selected_usds = []
        
        if mode == "HOMOGENEOUS":
            # Pick 1 random object type for all
            obj = random.choice(all_usd_paths)
            selected_usds = [obj] * num_active
            
        elif mode == "HOMO_80_20":
            # 80% Main Object, 20% Distractors
            main_obj = random.choice(all_usd_paths)
            distractors = [o for o in all_usd_paths if o != main_obj]
            
            count_main = int(num_active * 0.8)
            count_dist = num_active - count_main
            
            selected_usds = [main_obj] * count_main
            selected_usds += [random.choice(distractors) for _ in range(count_dist)]
            random.shuffle(selected_usds) # Shuffle so they aren't stacked in order
            
        elif mode == "CHAOS":
            # Pure random
            selected_usds = [random.choice(all_usd_paths) for _ in range(num_active)]
            
        selected_mats = []
        
        if mat_mode == "HOMOGENEOUS":
            # 1 Material for ALL parts
            mat = random.choice(available_materials)
            selected_mats = [mat] * num_active
            
        elif mat_mode == "HOMO_80_20":
            # 80% Main Material, 20% Distractor Materials
            main_mat = random.choice(available_materials)
            distractor_mats = [m for m in available_materials if m != main_mat]
            if not distractor_mats: distractor_mats = [main_mat]
            
            count_main = int(num_active * 0.8)
            count_dist = num_active - count_main
            
            selected_mats = [main_mat] * count_main
            selected_mats += [random.choice(distractor_mats) for _ in range(count_dist)]
            random.shuffle(selected_mats)
            
        elif mat_mode == "CHAOS":
            # Every part gets a random material
            selected_mats = [random.choice(available_materials) for _ in range(num_active)]

        # 3. Apply to Prims (Reference Swapping)
        for i, prim_path in enumerate(pool_paths):
            prim = stage.GetPrimAtPath(prim_path)
            
            if i < len(selected_usds):
                usd_to_load = selected_usds[i]
                
                # A. Swap Reference (This is fast!)
                refs = prim.GetReferences()
                refs.ClearReferences()
                refs.AddReference(usd_to_load)
                mat_path = selected_mats[i]
                mat_prim = stage.GetPrimAtPath(mat_path)
                #print(f"attempting to bind {mat_path} with {usd_to_load} to {prim.GetName()}")
                if mat_prim:
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                        UsdShade.Material(mat_prim),
                        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                        #materialPurpose="" 
                    )

                    
                        # B. Enable Physics & Vis
                UsdGeom.Imageable(prim).MakeVisible()
                UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(True)

                
                # # C. Randomize Scale/Material (Your Request)
                # self.randomize_single_prim_attributes(prim) 

            else:
                # --- INACTIVE PART ---
                # Disable Physics & Vis
                    
                        # B. Enable Physics & Vis
                #UsdGeom.Imageable(prim).MakeVisible()
                #UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(True)
                UsdGeom.Imageable(prim).MakeInvisible()
                UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(False)
                
                # Teleport away to be safe
                xform = UsdGeom.Xformable(prim)
                xform.ClearXformOpOrder()
                xform.AddTranslateOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(0.0, -10000.0, 0.0))
                
        return num_active # Return how many active parts were set, useful as first num_active in pool are active
                    
                
def get_bounds(prim_path):
    cache = bounds_utils.create_bbox_cache()
    return np.array(bounds_utils.compute_aabb(cache,prim_path,include_children=True))

def create_objects_json(asset_paths,output_path="./Config/objects.json"):
    """
    Searches asset folder (assumed to be in format)
    /assets
        --/bins
            -bin_1.usd

        --/materials
        --/parts
            -part_1.usd
        --stage.usd
        
    Produces a JSON file with the format:
    {
        "bins": {
            bin_1: {
                "name": "bin_1",
                "usd_filepath": "/path/to/bin_1.usd",
                "class": "0"
            },
        },
        "parts": {
            part_1: {
                "name": "part_1",
                "usd_filepath": "/path/to/part_1.usd",
                "class": "1"
            },
        }
        "stage": {
            "name": "stage",
            "usd_filepath": "/path/to/stage.usd"
        }
    }
    

    Args:
        asset_paths (_type_): path to /assets folder
        output_path (str, optional): output path for json file. Defaults to "./Config/objects.json".
    """
    
    assets = {
        "bins": {},
        "parts": {},
        "stage": {}
    }
    
    bins_path = os.path.join(asset_paths,"bins")
    parts_path = os.path.join(asset_paths,"parts")
    
    if not os.path.exists(asset_paths):
        print(f"Asset path {asset_paths} does not exist!")
        return
        
    
    #process bins
    if os.path.exists(bins_path):
        bin_files = [f for f in os.listdir(bins_path) if f.endswith('.usd')]
        for idx,bin_file in enumerate(bin_files):
            bin_label = f"bin_{idx}"
            bin_name = os.path.splitext(bin_file)[0]
            assets["bins"][bin_label] = {
                "name": bin_name,
                "usd_filepath": os.path.join(bins_path,bin_file),
                "class": "0"
            }
    else:
        print(f"No bins folder found at {bins_path}, skipping bin processing.")
    
    #process parts
    if os.path.exists(parts_path):
        part_files = [f for f in os.listdir(parts_path) if f.endswith('.usd')]
        for idx,part_file in enumerate(part_files):
            part_label = f"part_{idx}"
            part_name = os.path.splitext(part_file)[0]
            assets["parts"][part_label] = {
                "name": part_name,
                "usd_filepath": os.path.join(parts_path,part_file),
                "class": "1"
            }
    else:
        print(f"No parts folder found at {parts_path}, skipping part processing.")
    
    #process stage
    for file in os.listdir(asset_paths):
        if file.endswith('.usd') and 'stage' in file.lower():
            stage_path = os.path.join(asset_paths,file)
            stage_name = os.path.splitext(file)[0]
            assets["stage"] = {
                "name": stage_name,
                "usd_filepath": stage_path
            }
    
    #write to json
    with open(output_path,'w') as f:
        json.dump(assets,f,indent=4)
    
    print(f"Created objects.json at {output_path}")
    
    
    

# def get_position_from_voxel_index(voxel_index,voxel_size,grid_origin,jitter=0.0):
#     """
#     Given a voxel index (i,j,k), voxel size, and grid origin, compute the world position of the center of the voxel.
#     Adds optional random jitter to each axis within the range [-jitter, jitter].
#     """
#     i,j,k = voxel_index
#     x = grid_origin[0] + (i + 0.5) * voxel_size[0] + random.uniform(-jitter, jitter)
#     y = grid_origin[1] + (j + 0.5) * voxel_size[1] + random.uniform(-jitter, jitter)
#     z = grid_origin[2] + (k + 0.5) * voxel_size[2] + random.uniform(-jitter, jitter)
#     return (x,y,z)



def get_position_from_voxel_index(voxel_index, voxel_size, grid_origin, grid_counts, jitter=0.0):
    """
    grid_counts: A tuple (nx, ny, nz) representing the total number of voxels in each axis.
                 Used to center the grid around grid_origin for X and Y.
    """
    i, j, k = voxel_index
    nx, ny, _ = grid_counts  # Total count of voxels in X and Y (Z is unused for centering)

    # 1. Calculate the 'center offset'
    # We subtract half the total grid width from the origin to start 'left/back' of the center.
    # Formula: (index - (total_count / 2) + 0.5) centers the range.
    
    x = grid_origin[0] + (i - nx / 2.0 + 0.5) * voxel_size[0] + random.uniform(-jitter, jitter)
    y = grid_origin[1] + (j - ny / 2.0 + 0.5) * voxel_size[1] + random.uniform(-jitter, jitter)
    
    # Z usually stays 'floor-up' (stacking on top of origin), so we keep original logic.
    # If you want Z centered too, replace 'k' with '(k - nz / 2.0)'
    z = grid_origin[2] + (k + 0.5) * voxel_size[2] + random.uniform(-jitter, jitter)
    
    return (x, y, z)


def get_voxel_positions_vectorised(voxel_size, grid_origin, grid_counts, jitter=0.0):
    """
    Vectorised version to compute world positions for multiple voxel indices.
    grid_counts: A tuple (nx, ny, nz) representing the total number of voxels in each axis.
    Returns Nx3 numpy array of world positions, such that each index maps to a position (in local bin space).
    Voxel indices are un-needed as positions are computed for all voxels in the grid.
    """
    
    nx, ny, nz = grid_counts
    
    # Create meshgrid of all voxel indices
    i_indices, j_indices, k_indices = np.meshgrid(
        np.arange(nx), 
        np.arange(ny), 
        np.arange(nz),
        indexing='ij'
    )
    
    # Flatten to get all combinations
    i_flat = i_indices.flatten()
    j_flat = j_indices.flatten()
    k_flat = k_indices.flatten()
    
    # Compute positions using vectorised operations
    x = grid_origin[0] + (i_flat - nx / 2.0 + 0.5) * voxel_size[0]
    y = grid_origin[1] + (j_flat - ny / 2.0 + 0.5) * voxel_size[1]
    z = grid_origin[2] + (k_flat + 0.5) * voxel_size[2]
    
    # Add jitter if specified
    if jitter > 0.0:
        x += np.random.uniform(-jitter, jitter, size=x.shape)
        y += np.random.uniform(-jitter, jitter, size=y.shape)
        z += np.random.uniform(-jitter, jitter, size=z.shape)
    
    # Stack into N x 3 array
    positions = np.column_stack((x, y, z))
    
    return positions





#create_objects_json(asset_paths="/home/kaelin/BinPicking/SDG/IS/assets",output_path="./Config/objects.json")