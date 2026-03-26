import os
import sys

# Start Isaac Sim Simulation App headless
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.usd
from pxr import UsdGeom, Usd, Gf, Vt, Sdf
import isaacsim.core.utils.bounds as bounds_utils

asset_dir = "/home/kaelin/BinPicking/SDG/IS/assets"

def process_usd(path):
    print(f"Processing: {path}")
    omni.usd.get_context().open_stage(path)
    stage = omni.usd.get_context().get_stage()
    simulation_app.update()
    
    root_prim = stage.GetDefaultPrim()
    if not root_prim.IsValid():
        root_prim = list(stage.Traverse())[1]  # first child of absolute root /
    
    # Compute bounds in native un-transformed local space
    cache = bounds_utils.create_bbox_cache()
    bounds = cache.ComputeLocalBound(root_prim).ComputeAlignedRange()
    
    min_b = bounds.GetMin()
    max_b = bounds.GetMax()
    
    if min_b[0] == float('inf') or max_b[0] == float('-inf'):
        print(f"Skipping {path}, empty bounds.")
        return
        
    extents = Vt.Vec3fArray([
        Gf.Vec3f(float(min_b[0]), float(min_b[1]), float(min_b[2])), 
        Gf.Vec3f(float(max_b[0]), float(max_b[1]), float(max_b[2]))
    ])
    
    try:
        UsdGeom.ModelAPI.Apply(root_prim).SetExtentsHint(extents)
        stage.Save()
        print(f"Successfully baked bounds {extents} to {path}")
    except Exception as e:
        print(f"Failed configuring {path}: {e}")

# Process parts
parts_dir = os.path.join(asset_dir, "parts")
if os.path.exists(parts_dir):
    for f in os.listdir(parts_dir):
        if f.endswith(".usd"):
            process_usd(os.path.join(parts_dir, f))

# Process bins
bins_dir = os.path.join(asset_dir, "bins")
if os.path.exists(bins_dir):
    for d in os.listdir(bins_dir):
        d_path = os.path.join(bins_dir, d)
        if os.path.isdir(d_path):
            for f in os.listdir(d_path):
                if f.endswith(".usd"):
                    process_usd(os.path.join(d_path, f))

simulation_app.close()
print("Extents bake operation completed!")
