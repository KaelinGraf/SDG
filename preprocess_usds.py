from isaacsim import SimulationApp
import warp
simulation_app = SimulationApp({
    "headless": True,
}) 

import omni.usd
from pxr import Usd, UsdGeom, Gf, Sdf
import json
import os

# --- Copy of your UV generation logic ---
def generate_box_uvs(root_prim, scale=10.0):
    range_iterator = Usd.PrimRange(root_prim)
    for prim in range_iterator:
        if not prim.IsA(UsdGeom.Mesh):
            continue
            
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
        face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
        
        if not points or not face_vertex_indices:
            continue

        pv_api = UsdGeom.PrimvarsAPI(prim)
        # Check if UVs already exist to avoid double-processing
        if pv_api.HasPrimvar("st"): 
            print(f"Skipping {prim.GetPath()}, 'st' primvar already exists.")
            continue

        uvs = []
        idx_pointer = 0
        for count in face_vertex_counts:
            face_indices = face_vertex_indices[idx_pointer : idx_pointer + count]
            idx_pointer += count
            
            p0 = Gf.Vec3f(points[face_indices[0]])
            p1 = Gf.Vec3f(points[face_indices[1]])
            p2 = Gf.Vec3f(points[face_indices[2]])
            
            v1 = p1 - p0
            v2 = p2 - p0
            normal = Gf.Cross(v1, v2).GetNormalized()
            
            if abs(normal[0]) >= abs(normal[1]) and abs(normal[0]) >= abs(normal[2]):
                u_idx, v_idx = 1, 2
            elif abs(normal[1]) >= abs(normal[0]) and abs(normal[1]) >= abs(normal[2]):
                u_idx, v_idx = 0, 2
            else:
                u_idx, v_idx = 0, 1

            for v_idx_in_face in face_indices:
                p = points[v_idx_in_face]
                u = p[u_idx] * scale
                v = p[v_idx] * scale
                uvs.append(Gf.Vec2f(u, v))

        pv = pv_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
        pv.Set(uvs)
        print(f"Generated Box UVs for {prim.GetPath()}")

# --- Batch Processing Logic ---
def process_all_objects():
    # Load your objects config
    config_path = "./Config/objects.json"
    if not os.path.exists(config_path):
        print("Could not find objects.json")
        return

    with open(config_path, 'r') as f:
        objects_config = json.load(f)

    context = omni.usd.get_context()

    for obj_name, data in objects_config.items():
        usd_path = data["usd_filepath"]
        print(f"Processing {obj_name} at {usd_path}...")
        
        # Open the stage
        context.open_stage(usd_path)
        stage = context.get_stage()
        
        if not stage:
            print(f"Failed to open stage: {usd_path}")
            continue

        # Generate UVs on the default prim (and children)
        root_prim = stage.GetDefaultPrim()
        generate_box_uvs(root_prim)
        
        # Save the stage with the new UVs baked in
        stage.GetRootLayer().Save()
        print(f"Saved updates to {usd_path}")

# Run it
process_all_objects()