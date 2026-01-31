#provides convenience functions for mesh processing and manipulation
import isaacsim.core.utils.bounds as bounds_utils
import isaacsim.core.utils.prims as prim_utils
import numpy as np
import random
import os 
import json
from pathlib import Path


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
                prim.SetInstanceable(True)
                                              
                bounds = np.array(bounds_utils.compute_aabb(cache,prim_path))
                #compute the diagonal (eg 0,0,0) to (x,y,z)
                diag_vector = bounds[3:] - bounds[0:3]
                #print(f"diagonal vector: {diag_vector}")
                diag_length = np.linalg.norm(diag_vector) #this is the diameter of a bounding sphere around the object (max distance across the object)
                #store in asset registry
                #print(bounds)
                #print(f"diagonal length: {diag_length}")
                self.asset_registry[obj_name] = {
                    "bounds": bounds,
                    "diag_length": diag_length
                }
                prim_utils.set_prim_visibility(prim,False)
                #delete temp prim
                #prim_utils.delete_prim(temp_path)
                
                
def get_bounds(prim_path):
    cache = bounds_utils.create_bbox_cache()
    return np.array(bounds_utils.compute_aabb(cache,prim_path))

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





#create_objects_json(asset_paths="/home/kaelin/BinPicking/SDG/IS/assets",output_path="./Config/objects.json")