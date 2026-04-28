# patch_metadata.py
# Batch post-processes all distractor USDs in objects.json.
#
# Pipeline order (per asset):
#   0. Fix xformOp precision (GfQuatf → GfQuatd)
#   1. Get OBB → smart scale
#   2. Restructure: wrap in Xform, align + center inner geometry
#   3. Apply physics to wrapper (root)
#
# After patching, each USD has this hierarchy:
#   /Root    (wrapper Xform, identity transform — scene_builder manipulates this)
#     /Root/Geometry   (inner transform: centers COM at wrapper origin, aligns Y)
#       /Root/Geometry/mesh_0 ...
#
# This means: moving the wrapper to (x,y,z) places the part's COM at (x,y,z).
# The wrapper's Y axis is aligned with the part's longest axis.

import json
import os
import gc
import random
import argparse
import numpy as np

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf


CONFIG_PATH = "./Config/objects.json"


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def get_root_prim(stage):
    """Returns the root Xformable prim, or None."""
    root = stage.GetDefaultPrim()
    if not root:
        children = list(stage.GetPseudoRoot().GetChildren())
        if not children:
            return None
        root = children[0]
    return root if root.IsA(UsdGeom.Xformable) else None


def read_prim_pose(prim):
    """
    Reads [translate, orient, scale] from a prim's xformOps.
    Returns (Gf.Vec3d, Gf.Quatd, Gf.Vec3d).
    """
    translate = Gf.Vec3d(0, 0, 0)
    orient = Gf.Quatd(1, 0, 0, 0)
    scale = Gf.Vec3d(1, 1, 1)

    if prim and prim.IsA(UsdGeom.Xformable):
        xform = UsdGeom.Xformable(prim)
        for op in xform.GetOrderedXformOps():
            t = op.GetOpType()
            raw = op.Get()
            if raw is None:
                continue
            if t == UsdGeom.XformOp.TypeTranslate:
                translate = Gf.Vec3d(raw[0], raw[1], raw[2])
            elif t == UsdGeom.XformOp.TypeOrient:
                r = raw.GetReal()
                im = raw.GetImaginary()
                orient = Gf.Quatd(r, im[0], im[1], im[2])
            elif t == UsdGeom.XformOp.TypeScale:
                scale = Gf.Vec3d(raw[0], raw[1], raw[2])

    return translate, orient, scale


def set_prim_xform(prim, translate, orient, scale):
    """
    Clears a prim's xformOps, removes underlying attrs, recreates
    [translate, orient, scale] with PrecisionDouble.
    """
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    for attr_name in ["xformOp:translate", "xformOp:orient", "xformOp:scale"]:
        if prim.GetAttribute(attr_name).IsValid():
            prim.RemoveProperty(attr_name)
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(translate)
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(orient)
    xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(scale)


def get_world_vertices(stage):
    """Extracts all mesh vertices in world space. Returns np.ndarray (N,3)."""
    all_verts = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        if not points:
            continue
        xc = UsdGeom.XformCache(Usd.TimeCode.Default())
        wx = xc.GetLocalToWorldTransform(prim)
        for pt in points:
            p = wx.Transform(Gf.Vec3d(pt[0], pt[1], pt[2]))
            all_verts.append([p[0], p[1], p[2]])
    return np.array(all_verts) if all_verts else np.zeros((0, 3))


def compute_surface_centroid(stage):
    """
    Face-area-weighted centroid. Each triangle contributes proportionally
    to its surface area. Returns np.ndarray (3,) or None.
    """
    weighted_sum = np.zeros(3)
    total_area = 0.0

    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        fvc = mesh.GetFaceVertexCountsAttr().Get()
        fvi = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not fvc or not fvi:
            continue

        xc = UsdGeom.XformCache(Usd.TimeCode.Default())
        wx = xc.GetLocalToWorldTransform(prim)
        wpts = np.array([[*wx.Transform(Gf.Vec3d(p[0], p[1], p[2]))] for p in points])

        idx = 0
        for count in fvc:
            if count < 3:
                idx += count
                continue
            v0 = wpts[fvi[idx]]
            for tri in range(count - 2):
                v1 = wpts[fvi[idx + tri + 1]]
                v2 = wpts[fvi[idx + tri + 2]]
                tc = (v0 + v1 + v2) / 3.0
                area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
                weighted_sum += tc * area
                total_area += area
            idx += count

    if total_area < 1e-12:
        return None
    return weighted_sum / total_area


def compute_obb(stage):
    """
    OBB via PCA on world-space vertices.
    Returns (eigenvectors_3x3, half_extents_3, longest_axis_unit_vec) or None.
    """
    verts = get_world_vertices(stage)
    if len(verts) < 4:
        return None

    centroid = verts.mean(axis=0)
    centered = verts - centroid

    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]

    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] *= -1

    projected = centered @ eigenvectors
    half_extents = (projected.max(axis=0) - projected.min(axis=0)) / 2.0

    longest_idx = np.argmax(half_extents)
    longest_axis = eigenvectors[:, longest_idx]

    return eigenvectors, half_extents, longest_axis


# ═════════════════════════════════════════════════════════════════════════════
# Step 0: Fix xformOp precision (GfQuatf → GfQuatd)
# ═════════════════════════════════════════════════════════════════════════════

def fix_xform_ops_to_double_precision(stage):
    """
    Converts all xformOps on every prim to double precision.
    Also sets stage metadata (meters, Z-up).
    """
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if not prim.IsA(UsdGeom.Xformable):
            continue

        xformable = UsdGeom.Xformable(prim)
        existing_ops = xformable.GetOrderedXformOps()
        if not existing_ops:
            continue

        translate_val = None
        orient_val = None
        scale_val = None
        op_attr_names = []

        for op in existing_ops:
            op_attr_names.append(op.GetAttr().GetName())
            op_type = op.GetOpType()
            if op_type == UsdGeom.XformOp.TypeTranslate:
                raw = op.Get()
                if raw is not None:
                    translate_val = Gf.Vec3d(raw[0], raw[1], raw[2])
            elif op_type == UsdGeom.XformOp.TypeOrient:
                raw = op.Get()
                if raw is not None:
                    r = raw.GetReal()
                    im = raw.GetImaginary()
                    orient_val = Gf.Quatd(r, im[0], im[1], im[2])
            elif op_type == UsdGeom.XformOp.TypeScale:
                raw = op.Get()
                if raw is not None:
                    scale_val = Gf.Vec3d(raw[0], raw[1], raw[2])

        xformable.ClearXformOpOrder()
        for attr_name in op_attr_names:
            prim.RemoveProperty(attr_name)

        if translate_val is not None:
            xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(translate_val)
        if orient_val is not None:
            xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(orient_val)
        else:
            xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
        if scale_val is not None:
            xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(scale_val)

    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)


# ═════════════════════════════════════════════════════════════════════════════
# Step 1: Smart scale (OBB-based, shape-aware)
# ═════════════════════════════════════════════════════════════════════════════

MAX_GRIPPER_STROKE = 0.140   # 140mm
MIN_VISIBLE_DIM    = 0.008   # 8mm
MAX_BIN_DIM        = 0.300   # 300mm

def rescale_smart(stage):
    """
    STEP 1: Get OBB → classify shape → apply uniform scale to root prim.
    Returns (scale_factor, shape_class, final_dims_sorted) or None.
    """
    obb = compute_obb(stage)
    if obb is None:
        return None
    _, half_extents, _ = obb

    dims = np.sort(half_extents * 2)  # full extents, ascending
    d_small, d_mid, d_long = dims[0], dims[1], dims[2]

    if d_long < 1e-8:
        return None

    ar = d_long / max(d_small, 1e-8)

    if ar < 2.0:
        sf = random.uniform(0.030, 0.080) / d_long
    elif ar < 5.0:
        sf = random.uniform(0.015, 0.050) / d_small
    else:
        sf = random.uniform(0.010, 0.030) / d_small

    # Hard clamps
    if d_long * sf > MAX_BIN_DIM:
        sf = MAX_BIN_DIM / d_long
    if d_small * sf < MIN_VISIBLE_DIM:
        sf = MIN_VISIBLE_DIM / d_small
    if d_small * sf > MAX_GRIPPER_STROKE:
        sf = MAX_GRIPPER_STROKE / d_small

    # Apply: multiply existing scale on root prim
    root = get_root_prim(stage)
    if not root:
        return None
    _, _, old_scale = read_prim_pose(root)
    new_scale = Gf.Vec3d(old_scale[0] * sf, old_scale[1] * sf, old_scale[2] * sf)
    _, orient, _ = read_prim_pose(root)
    translate, _, _ = read_prim_pose(root)
    set_prim_xform(root, translate, orient, new_scale)

    shape_class = "COMPACT" if ar < 2.0 else ("ELONGATED" if ar < 5.0 else "VERY_LONG")
    return sf, shape_class, dims * sf


# ═════════════════════════════════════════════════════════════════════════════
# Step 2: Restructure hierarchy + align + center
# ═════════════════════════════════════════════════════════════════════════════

def restructure_and_align(stage):
    """
    STEP 2: Wraps the existing root in a parent Xform, then sets the inner
    prim's transform so that:
      - The wrapper's origin = part's COM  (face-area-weighted)
      - The wrapper's Y axis  = part's longest OBB axis
    
    IDEMPOTENT: If the root already has a "Geometry" child (from a previous
    run), it reuses the existing wrapper/inner structure instead of re-wrapping.
    
    After this step the USD hierarchy is:
      /OrigName       (wrapper Xform, identity — scene_builder manipulates this)
        /OrigName/Geometry  (inner: centering + alignment + scale baked in)
          /OrigName/Geometry/mesh_0 ...
    
    Returns True on success.
    """
    # ── 2a. Get root and check if already restructured ──
    old_root = get_root_prim(stage)
    if not old_root:
        return False
    old_path = old_root.GetPath()

    # Check for existing Geometry child → already wrapped
    geom_path = old_path.AppendChild("Geometry")
    existing_inner = stage.GetPrimAtPath(geom_path)
    already_wrapped = existing_inner and existing_inner.IsValid() and existing_inner.IsA(UsdGeom.Xformable)

    if already_wrapped:
        # ── Already wrapped: reuse existing structure ──
        wrapper_prim = old_root
        inner_prim = existing_inner
        layer = stage.GetRootLayer()

        # ── Compute cumulative scale from the ENTIRE chain ──
        # wrapper_scale * inner_scale * intermediate_scale_1 * ... * deepest_scale
        # This preserves rescale_smart's contribution (on wrapper) and any
        # intermediate scales from previous buggy runs.
        _, _, wrapper_scale = read_prim_pose(wrapper_prim)
        cumulative_scale = [wrapper_scale[0], wrapper_scale[1], wrapper_scale[2]]

        cursor = inner_prim
        while True:
            _, _, s = read_prim_pose(cursor)
            cumulative_scale[0] *= s[0]
            cumulative_scale[1] *= s[1]
            cumulative_scale[2] *= s[2]
            nested_path = cursor.GetPath().AppendChild("Geometry")
            nested = stage.GetPrimAtPath(nested_path)
            if nested and nested.IsValid() and nested.IsA(UsdGeom.Xformable):
                cursor = nested
            else:
                break
        deepest = cursor

        # ── Flatten nested Geometry chains if they exist ──
        if deepest.GetPath() != inner_prim.GetPath():
            print(f"    [FIX] Flattening nested Geometry: {deepest.GetPath()} → {inner_prim.GetPath()}")
            edit = Sdf.BatchNamespaceEdit()
            for child in deepest.GetChildren():
                child_name = child.GetName()
                target = inner_prim.GetPath().AppendChild(child_name)
                if stage.GetPrimAtPath(target).IsValid() and target != child.GetPath():
                    child_name = f"{child_name}_flat"
                    target = inner_prim.GetPath().AppendChild(child_name)
                edit.Add(child.GetPath(), target)
            if layer.Apply(edit):
                # Delete now-empty intermediate Geometry prims
                del_cursor = deepest.GetPath()
                while del_cursor != inner_prim.GetPath():
                    p = stage.GetPrimAtPath(del_cursor)
                    if p and p.IsValid() and len(p.GetChildren()) == 0:
                        stage.RemovePrim(del_cursor)
                    del_cursor = del_cursor.GetParentPath()

        existing_scale = Gf.Vec3d(cumulative_scale[0], cumulative_scale[1], cumulative_scale[2])
    else:
        # ── First time: restructure the hierarchy ──
        _, _, existing_scale = read_prim_pose(old_root)
        layer = stage.GetRootLayer()

        # Rename old root out of the way
        temp_path = Sdf.Path("/__temp_inner")
        edit1 = Sdf.BatchNamespaceEdit()
        edit1.Add(old_path, temp_path)
        if not layer.Apply(edit1):
            print("    [WARN] Could not rename root for restructure")
            return False

        # Create the wrapper Xform at the original path
        wrapper_xform = UsdGeom.Xform.Define(stage, old_path)
        wrapper_prim = wrapper_xform.GetPrim()
        stage.SetDefaultPrim(wrapper_prim)

        # Move old root under wrapper as "Geometry"
        inner_path = old_path.AppendChild("Geometry")
        edit2 = Sdf.BatchNamespaceEdit()
        edit2.Add(temp_path, inner_path)
        if not layer.Apply(edit2):
            print("    [WARN] Could not move inner geometry under wrapper")
            return False

        inner_prim = stage.GetPrimAtPath(inner_path)
        if not inner_prim or not inner_prim.IsValid():
            return False

    # ── 2b. Set wrapper to identity ──
    set_prim_xform(wrapper_prim,
                   Gf.Vec3d(0, 0, 0),
                   Gf.Quatd(1, 0, 0, 0),
                   Gf.Vec3d(1, 1, 1))

    # ── 2c. Reset inner prim to scale-only (identity T, identity R, keep S) ──
    set_prim_xform(inner_prim,
                   Gf.Vec3d(0, 0, 0),
                   Gf.Quatd(1, 0, 0, 0),
                   existing_scale)

    # ── 2d. Compute OBB on the now-clean geometry (world verts = scale * local) ──
    verts = get_world_vertices(stage)
    if len(verts) < 4:
        return False

    centroid_v = verts.mean(axis=0)
    centered = verts - centroid_v
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] *= -1

    projected = centered @ eigenvectors
    half_extents = (projected.max(axis=0) - projected.min(axis=0)) / 2.0
    longest_idx = np.argmax(half_extents)
    longest_axis = eigenvectors[:, longest_idx]

    # ── 2e. Compute R_align: maps longest OBB axis → Y ──
    target = np.array([0.0, 1.0, 0.0])
    dot = np.clip(np.dot(longest_axis, target), -1.0, 1.0)
    cross = np.cross(longest_axis, target)
    cross_norm = np.linalg.norm(cross)

    if cross_norm < 1e-6:
        if dot > 0:
            r_align = Gf.Rotation(Gf.Vec3d(0, 0, 1), 0)  # identity
        else:
            r_align = Gf.Rotation(Gf.Vec3d(0, 0, 1), 180)
    else:
        ax = cross / cross_norm
        ang = float(np.degrees(np.arccos(dot)))
        r_align = Gf.Rotation(
            Gf.Vec3d(float(ax[0]), float(ax[1]), float(ax[2])), ang
        )

    # ── 2f. Compute face-area-weighted COM (in scaled-mesh space) ──
    com = compute_surface_centroid(stage)
    if com is None:
        com = verts.mean(axis=0)  # fallback to vertex mean

    # ── 2g. Compute inner translate ──
    #   T = -R * COM_scaled
    com_gf = Gf.Vec3d(float(com[0]), float(com[1]), float(com[2]))
    rot_mat = Gf.Matrix4d(1.0)
    rot_mat.SetRotateOnly(r_align)
    rotated_com = rot_mat.TransformDir(com_gf)
    inner_translate = Gf.Vec3d(-rotated_com[0], -rotated_com[1], -rotated_com[2])

    # Convert R_align to quaternion
    rq = r_align.GetQuat()
    inner_orient = Gf.Quatd(
        rq.GetReal(),
        rq.GetImaginary()[0],
        rq.GetImaginary()[1],
        rq.GetImaginary()[2],
    )

    # ── 2h. Apply the final inner transform ──
    set_prim_xform(inner_prim, inner_translate, inner_orient, existing_scale)

    return True


# ═════════════════════════════════════════════════════════════════════════════
# Step 3: Physics (on wrapper / root)
# ═════════════════════════════════════════════════════════════════════════════

def apply_physics(stage):
    """
    Applies physics APIs with correct prim-type separation:
      - Wrapper (root Xform): RigidBody + Mass + PhysX tuning only
      - Descendant Mesh prims: CollisionAPI + MeshCollisionAPI (convexHull)
      - Non-Mesh descendants (e.g. Geometry Xform): strip any pre-existing physics
    """
    root = get_root_prim(stage)
    if not root:
        return

    # ── Strip pre-existing physics from ALL non-Mesh descendants ──
    # This prevents the Geometry Xform from having stale physics APIs
    # that cause "invalid inertia tensor" errors.
    physics_apis_to_strip = [
        UsdPhysics.RigidBodyAPI,
        UsdPhysics.MassAPI,
        UsdPhysics.CollisionAPI,
        UsdPhysics.MeshCollisionAPI,
    ]
    for prim in Usd.PrimRange(root):
        if prim == root:
            continue  # handle root separately below
        if prim.IsA(UsdGeom.Mesh):
            continue  # meshes get collision below
        # Strip physics from non-mesh descendants (Xform, Scope, etc.)
        for api_type in physics_apis_to_strip:
            if prim.HasAPI(api_type):
                prim.RemoveAPI(api_type)
        if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)

    # ── Root (wrapper Xform): RigidBody + Mass + PhysX tuning ONLY ──
    # NO CollisionAPI on root — collision lives on the actual Mesh prims.

    if not root.HasAPI(UsdPhysics.RigidBodyAPI):
        rb = UsdPhysics.RigidBodyAPI.Apply(root)
    else:
        rb = UsdPhysics.RigidBodyAPI(root)
    rb.CreateRigidBodyEnabledAttr(True)
    rb.CreateKinematicEnabledAttr(False)

    if not root.HasAPI(UsdPhysics.MassAPI):
        ma = UsdPhysics.MassAPI.Apply(root)
    else:
        ma = UsdPhysics.MassAPI(root)
    ma.CreateDensityAttr(2710.0)

    # Remove CollisionAPI from root if it exists (it's an Xform, not a Mesh)
    if root.HasAPI(UsdPhysics.CollisionAPI):
        root.RemoveAPI(UsdPhysics.CollisionAPI)
    if root.HasAPI(UsdPhysics.MeshCollisionAPI):
        root.RemoveAPI(UsdPhysics.MeshCollisionAPI)

    if not root.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
        px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
        px.CreateEnableCCDAttr(True)
        px.CreateSolverPositionIterationCountAttr(8)
        px.CreateSolverVelocityIterationCountAttr(0)
        px.CreateSleepThresholdAttr(0.005)
        px.CreateLinearDampingAttr(0.5)
        px.CreateAngularDampingAttr(0.5)
        px.CreateMaxLinearVelocityAttr(5.0)
        px.CreateMaxDepenetrationVelocityAttr(1.0)

    # ── Descendant Mesh prims: Collision + convexHull ──
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            ca = UsdPhysics.CollisionAPI.Apply(prim)
            ca.CreateCollisionEnabledAttr(True)
        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            mc = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mc.CreateApproximationAttr("convexHull")


# ═════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═════════════════════════════════════════════════════════════════════════════

def patch_all(config_path=CONFIG_PATH, align_y=True):
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found.")
        return

    with open(config_path, 'r') as f:
        objects_config = json.load(f)

    parts = list(objects_config.get("parts", {}).items())
    total = len(parts)
    stats = {"patched": 0, "skipped": 0, "failed": 0}
    purge_keys = []

    print(f"Patching {total} USD assets...")
    print(f"  0. Fix xformOp precision → GfQuatd")
    print(f"  1. OBB → smart scale (gripper≤{MAX_GRIPPER_STROKE*1000:.0f}mm, "
          f"visible≥{MIN_VISIBLE_DIM*1000:.0f}mm, bin≤{MAX_BIN_DIM*1000:.0f}mm)")
    print(f"  2. Wrap in Xform → align longest→Y, center COM at origin")
    print(f"  3. Physics: RigidBody + Collision(convexHull) + Mass(2710)")
    print()

    for i, (part_key, part_data) in enumerate(parts):
        usd_path = part_data.get("usd_filepath")
        if not usd_path or not os.path.exists(usd_path):
            stats["skipped"] += 1
            purge_keys.append((part_key, usd_path))
            print(f"  [{i}/{total}] SKIP (missing): {part_key}")
            continue

        try:
            stage = Usd.Stage.Open(usd_path)
            if not stage:
                stats["skipped"] += 1
                purge_keys.append((part_key, usd_path))
                print(f"  [{i}/{total}] SKIP (can't open): {part_key}")
                continue

            # Step 0: Fix precision
            fix_xform_ops_to_double_precision(stage)

            # Step 1: OBB → smart scale
            scale_info = ""
            scale_result = rescale_smart(stage)
            if scale_result:
                sf, shape_class, fdims = scale_result
                scale_info = f"{shape_class} {fdims[0]*1000:.0f}×{fdims[1]*1000:.0f}×{fdims[2]*1000:.0f}mm"

            # Step 2: Restructure + align + center
            if align_y:
                ok = restructure_and_align(stage)
                if not ok:
                    print(f"  [{i}/{total}] WARN restructure failed: {part_key}")

            # Step 3: Physics
            apply_physics(stage)

            # Save
            stage.GetRootLayer().Save()
            stats["patched"] += 1

            if i % 50 == 0:
                print(f"  [{i}/{total}] OK: {part_key} | {scale_info}")
                gc.collect()

        except Exception as e:
            stats["failed"] += 1
            purge_keys.append((part_key, usd_path))
            print(f"  [{i}/{total}] FAILED {part_key}: {e}")

    # Purge bad entries
    if purge_keys:
        print(f"\nPurging {len(purge_keys)} bad entries...")
        for part_key, usd_path in purge_keys:
            if part_key in objects_config["parts"]:
                del objects_config["parts"][part_key]
            if usd_path and os.path.exists(usd_path):
                os.remove(usd_path)
                print(f"  Deleted: {usd_path}")
            if usd_path:
                sidecar = usd_path.replace(".usd", "")
                if os.path.isdir(sidecar):
                    import shutil
                    shutil.rmtree(sidecar, ignore_errors=True)

        with open(config_path, 'w') as f:
            json.dump(objects_config, f, indent=4)
        print(f"  Config saved ({len(objects_config['parts'])} parts remaining)")

    print(f"\n{'='*60}")
    print(f"Patch complete!")
    print(f"  Patched:  {stats['patched']}")
    print(f"  Skipped:  {stats['skipped']} (purged)")
    print(f"  Failed:   {stats['failed']} (purged)")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch distractor USD physics and geometry")
    parser.add_argument('--config', type=str, default=CONFIG_PATH,
                        help='Path to objects config JSON')
    parser.add_argument('--no-align-y', action='store_true', default=False,
                        help='Disable longest-axis-to-Y alignment')
    args = parser.parse_args()

    patch_all(config_path=args.config, align_y=not args.no_align_y)
    simulation_app.close()