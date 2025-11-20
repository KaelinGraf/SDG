#provides convenience functions for mesh processing and manipulation
import isaacsim.core.utils.bounds as bounds_utils
import isaacsim.core.utils.prims as prim_utils
import numpy as np
import random


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
        temp_path = "/World/Temp_Asset_Manager_Prim"
        cache = bounds_utils.create_bbox_cache()

        for obj_name, obj_info in self.objects_config.items():
            usd_filepath = obj_info.get("usd_filepath", None)
            
            if usd_filepath:
                prim = prim_utils.create_prim(
                    prim_path=temp_path,
                    usd_path=usd_filepath,
                )
                                              
                bounds = np.array(bounds_utils.compute_aabb(cache,temp_path))
                #compute the diagonal (eg 0,0,0) to (x,y,z)
                diag_vector = bounds[3:] - bounds[0:3]
                print(f"diagonal vector: {diag_vector}")
                diag_length = np.linalg.norm(diag_vector) #this is the diameter of a bounding sphere around the object (max distance across the object)
                #store in asset registry
                print(bounds)
                print(f"diagonal length: {diag_length}")
                self.asset_registry[obj_name] = {
                    "bounds": bounds,
                    "diag_length": diag_length
                }
                #delete temp prim
                prim_utils.delete_prim(temp_path)
    

def get_position_from_voxel_index(voxel_index,voxel_size,grid_origin,jitter=0.0):
    """
    Given a voxel index (i,j,k), voxel size, and grid origin, compute the world position of the center of the voxel.
    Adds optional random jitter to each axis within the range [-jitter, jitter].
    """
    i,j,k = voxel_index
    x = grid_origin[0] + (i + 0.5) * voxel_size[0] + random.uniform(-jitter, jitter)
    y = grid_origin[1] + (j + 0.5) * voxel_size[1] + random.uniform(-jitter, jitter)
    z = grid_origin[2] + (k + 0.5) * voxel_size[2] + random.uniform(-jitter, jitter)
    return (x,y,z)