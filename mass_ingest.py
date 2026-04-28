# mass_ingest.py
# Downloads graspable-object 3D assets from Objaverse, converts to USD,
# applies post-processing fixes (xform precision, physics stripping, scaling),
# and presents each one for manual visual approval before accepting.

import os
import json
import random
import asyncio
import argparse

# 1. Start Simulation App FIRST (Required before importing pxr or omni)
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni
from isaacsim.core.utils.extensions import enable_extension
enable_extension("omni.kit.asset_converter")

from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf, PhysxSchema
import omni.kit.asset_converter

try:
    import objaverse
except ImportError:
    print("Please install objaverse: ./python.sh -m pip install objaverse")
    exit()

try:
    import matplotlib
    matplotlib.use("TkAgg")  # Force interactive backend
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    print("[WARNING] matplotlib not found. Manual verification will use text-only mode.")
    HAS_MATPLOTLIB = False


# ─────────────────────────────────────────────────────────────────────────────
# Keyword sets for Objaverse filtering
# ─────────────────────────────────────────────────────────────────────────────

# Objects that could realistically appear in an industrial bin-picking cell.
# These are SPECIFIC terms — no generic words like "model", "part", "item" etc.
# that would match everything on Objaverse.
GRASPABLE_KEYWORDS = {
    # Fasteners
    "bolt", "screw", "nut", "washer", "rivet", "stud", "dowel",
    # Hardware / mechanical
    "bracket", "bushing", "bearing", "coupling", "fitting", "flange",
    "connector", "adapter", "gasket", "spacer", "standoff",
    "spring", "hinge", "latch", "clamp", "clip", "retainer",
    "fastener", "anchor", "eyelet", "grommet",
    # Mechanical drivetrain
    "gear", "sprocket", "pulley", "cam", "rotor", "impeller",
    # Pipe/tube fittings
    "valve", "nozzle", "elbow", "tee fitting", "pipe fitting",
    # Tools (specific)
    "wrench", "pliers", "screwdriver", "socket wrench", "hex key",
    # Industrial / CAD
    "machined", "cnc", "turned", "milled", "cast", "forged",
    "hydraulic", "pneumatic", "manifold",
    # Electrical hardware
    "terminal", "relay", "contactor", "circuit breaker",
    # Knobs and controls
    "knob", "lever", "dial", "switch",
    # Seals
    "o-ring", "seal", "retaining ring",
    # Small containers (industrial)
    "bottle", "jar", "canister", "vial",
    # Stationery
    "pen", "pencil", "marker", "eraser", "stapler",
    # Explicit CAD/engineering terms
    "cad", "engineering", "mechanical", "industrial",
    "fuselage", "mounting", "fixture", "jig",
}

# Absolutely reject anything matching these — art, game, organic, architectural
REJECT_KEYWORDS = {
    # Environments & architecture
    "scene", "room", "architecture", "building", "city", "map",
    "landscape", "terrain", "mountain", "tree", "forest", "garden",
    "house", "wall", "floor", "ceiling", "door", "window", "roof",
    "environment", "world", "level", "dungeon", "cave", "island",
    "bridge", "tower", "castle", "temple", "church", "ruins",
    # Characters & organic
    "character", "rigged", "animated", "skeleton", "armature",
    "animal", "human", "person", "body", "head", "face", "hand",
    "creature", "monster", "dragon", "zombie", "alien",
    "boyfriend", "girlfriend",
    # Vehicles & military
    "vehicle", "car", "truck", "tank", "aircraft", "plane", "helicopter",
    "ship", "boat", "submarine", "train", "locomotive", "motorcycle",
    "turret", "cannon", "missile", "weapon", "gun", "rifle", "sword",
    "armor", "shield", "military",
    # Furniture (too large for bin picking)
    "furniture", "chair", "table", "desk", "sofa", "bed", "shelf",
    "cabinet", "wardrobe", "bookcase", "couch", "bench",
    # Food & organic
    "food", "fruit", "vegetable", "meat", "cake", "bread", "apple",
    # Nature
    "plant", "flower", "grass", "rock", "stone", "cliff",
    # Clothing
    "clothing", "shirt", "pants", "dress", "hat", "shoe", "boot",
    # Art & sculpture
    "sculpture", "abstract", "art ", "artwork", "statue", "bust",
    "figurine", "miniature", "diorama", "vase",
    # Gaming & fantasy
    "game", "minecraft", "roblox", "pokemon", "anime", "cartoon",
    "fantasy", "sci-fi", "star gate", "magic", "spell",
    "fnf", "friday night", "fortnite", "among us",
    "low poly", "lowpoly", "high poly", "hipoly",
    # Photogrammetry scans (usually not clean geometry)
    "photogrammetry", "3d scan", "scan",
    # Musical
    "guitar", "piano", "drum", "keyboard", "instrument",
    # Jewelry
    "ring", "necklace", "bracelet", "earring", "jewelry", "jewel",
    "engagement",
    # Misc junk
    "covid", "virus", "chest", "treasure", "claw", "potion",
    "wand", "staff", "crown", "throne",
}

# Maximum faces for physics-safe simulation
MAX_FACE_COUNT = 20000
MIN_FACE_COUNT = 100  # Reject trivially simple shapes


class MassIngestionPipeline:
    def __init__(self, config_path="./Config/objects.json", output_dir="./assets/distractors/", align_z=True):
        self.config_path = config_path
        self.output_dir = output_dir
        self.align_z = align_z
        os.makedirs(self.output_dir, exist_ok=True)

        # Load existing config to append to
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.objects_config = json.load(f)
        else:
            self.objects_config = {"parts": {}}

        # Ensure "parts" key exists
        if "parts" not in self.objects_config:
            self.objects_config["parts"] = {}

        self.converter_context = omni.kit.asset_converter.AssetConverterContext()
        self.converter_context.ignore_camera = True
        self.converter_context.ignore_light = True
        self.converter_context.ignore_animation = True
        self.converter_context.export_preview_surface = True

        self.stats = {"downloaded": 0, "converted": 0, "accepted": 0, "rejected_auto": 0, "rejected_manual": 0}

    def download_objaverse_assets(self, num_assets=50):
        """Filters Objaverse for graspable, bin-picking-suitable objects and downloads them."""
        print("Loading Objaverse annotations (this takes a moment)...")
        annotations = objaverse.load_annotations()

        valid_uids = []

        for uid, ann in annotations.items():
            # Collect all text metadata
            tags = {t['name'].lower() for t in ann.get('tags', [])}
            categories = {c['name'].lower() for c in ann.get('categories', [])}
            name = ann.get('name', '').lower()
            description = ann.get('description', '').lower()
            all_words = tags.union(categories)
            all_text = ' '.join(all_words) + ' ' + name + ' ' + description

            # ── Reject pass (check first, cheaper) ──
            if any(kw in all_text for kw in REJECT_KEYWORDS):
                continue

            # ── Accept pass ──
            has_graspable = any(kw in all_text for kw in GRASPABLE_KEYWORDS)
            if not has_graspable:
                continue

            # ── Face count filter (metadata-based pre-filter) ──
            face_count = ann.get('faceCount', 0)
            if face_count < MIN_FACE_COUNT or face_count > MAX_FACE_COUNT:
                continue

            # ── Reject animated/rigged at metadata level ──
            if ann.get('animationCount', 0) > 0:
                continue

            valid_uids.append(uid)

        print(f"Found {len(valid_uids)} valid graspable objects (from {len(annotations)} total).")

        # Shuffle and sample
        random.shuffle(valid_uids)
        selected_uids = valid_uids[:min(num_assets, len(valid_uids))]

        print(f"Downloading {len(selected_uids)} assets (with retry on failure)...")
        objects = {}
        for i, uid in enumerate(selected_uids):
            for attempt in range(3):  # Up to 3 retries per asset
                try:
                    result = objaverse.load_objects(uids=[uid])
                    if result:
                        objects.update(result)
                        print(f"  Downloaded {len(objects)} / {len(selected_uids)} objects")
                    break
                except Exception as e:
                    if attempt < 2:
                        print(f"  [RETRY] Download failed for {uid} (attempt {attempt+1}/3): {e}")
                        import time
                        time.sleep(2)
                    else:
                        print(f"  [SKIP] Download failed for {uid} after 3 attempts: {e}")

        self.stats["downloaded"] = len(objects)
        return objects

    async def convert_to_usd(self, source_path, target_usd_path):
        """Converts OBJ/GLB/FBX to USD."""
        def progress_callback(progress, total_steps):
            pass

        converter = omni.kit.asset_converter.get_instance()
        task = converter.create_converter_task(
            source_path, target_usd_path, progress_callback, self.converter_context
        )
        success = await task.wait_until_finished()
        return success

    # ─────────────────────────────────────────────────────────────────────────
    # Post-processing methods
    # ─────────────────────────────────────────────────────────────────────────

    def fix_xform_ops_to_double_precision(self, stage):
        """
        FIX #1: Converts all xformOps to double precision.
        The asset converter creates ops with float precision (GfQuatf),
        but scene_builder writes GfQuatd to xformOp:orient at runtime.
        
        ClearXformOpOrder() only clears the ordering metadata — the underlying
        attributes (xformOp:orient, etc.) persist with their original types.
        We must explicitly RemoveProperty() each one before recreating with
        double precision.
        """
        for prim in Usd.PrimRange(stage.GetPseudoRoot()):
            if not prim.IsA(UsdGeom.Xformable):
                continue

            xformable = UsdGeom.Xformable(prim)
            existing_ops = xformable.GetOrderedXformOps()

            if not existing_ops:
                continue

            # Read current values and collect attribute names before clearing
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

            # 1. Clear the xformOpOrder metadata
            xformable.ClearXformOpOrder()

            # 2. Remove the actual underlying attributes so they can be
            #    recreated with a different type/precision
            for attr_name in op_attr_names:
                prim.RemoveProperty(attr_name)

            # 3. Recreate with double precision
            if translate_val is not None:
                new_t = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
                new_t.Set(translate_val)

            if orient_val is not None:
                new_o = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
                new_o.Set(orient_val)
            else:
                # Always create an orient op so scene_builder can write to it
                new_o = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
                new_o.Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))  # Identity

            if scale_val is not None:
                new_s = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
                new_s.Set(scale_val)

    def strip_physics_properties(self, stage):
        """
        FIX #2: Removes ALL pre-existing physics APIs from the USD.
        scene_builder.assign_physics_materials() re-applies physics at spawn time
        with controlled collision approximation (convexHull), so any embedded
        physics from the source model only causes problems.
        """
        physics_api_types = [
            UsdPhysics.RigidBodyAPI,
            UsdPhysics.CollisionAPI,
            UsdPhysics.MeshCollisionAPI,
            UsdPhysics.MassAPI,
        ]

        physx_schema_types = [
            "PhysxRigidBodyAPI",
            "PhysxCollisionAPI",
            "PhysxMeshCollisionAPI",
            "PhysxConvexHullCollisionAPI",
            "PhysxTriangleMeshCollisionAPI",
            "PhysxConvexDecompositionCollisionAPI",
        ]

        for prim in Usd.PrimRange(stage.GetPseudoRoot()):
            # Remove standard UsdPhysics APIs
            for api_type in physics_api_types:
                if prim.HasAPI(api_type):
                    prim.RemoveAPI(api_type)

            # Remove PhysX-specific schemas
            for schema_name in physx_schema_types:
                try:
                    schema_type = getattr(PhysxSchema, schema_name, None)
                    if schema_type and prim.HasAPI(schema_type):
                        prim.RemoveAPI(schema_type)
                except Exception:
                    pass  # Schema type may not exist in this USD version

            # Remove physics-related attributes that may linger
            for attr_name in ["physics:rigidBodyEnabled", "physics:collisionEnabled",
                              "physics:mass", "physics:density",
                              "physxRigidBody:linearDamping", "physxRigidBody:angularDamping"]:
                attr = prim.GetAttribute(attr_name)
                if attr and attr.IsValid():
                    prim.RemoveProperty(attr_name)

    def normalize_geometry(self, stage, align_z=True):
        """
        Normalizes the object geometry:
        1. Sets stage metadata (Z-up, meters)
        2. Checks aspect ratio (rejects extreme shapes)
        3. Centers the object at origin
        4. Optionally aligns the longest axis to Z (for consistent pose extraction)
        5. Scales to graspable size (5-25cm)
        
        MUST be called AFTER fix_xform_ops_to_double_precision.
        Clears the root prim's xformOps and rebuilds them with the composite transform.
        
        Returns scale_factor on success, or None to signal rejection.
        """
        # 1. Fix Core Stage Metadata
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

        # 2. Compute the current bounding box
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        bounds = bbox_cache.ComputeWorldBound(stage.GetPseudoRoot())
        box_range = bounds.ComputeAlignedRange()

        size = box_range.GetSize()
        dims = [size[0], size[1], size[2]]
        max_dim = max(dims)
        positive_dims = [s for s in dims if s > 1e-6]
        if not positive_dims:
            return None
        min_dim = min(positive_dims)

        # 3. Log aspect ratio (info only — manual review decides)
        if min_dim > 1e-6:
            aspect_ratio = max_dim / min_dim
            print(f"  Aspect ratio: {aspect_ratio:.1f}:1")

        if max_dim <= 0:
            return None

        # 4. Find root prim
        root_prim = stage.GetDefaultPrim()
        if not root_prim:
            children = list(stage.GetPseudoRoot().GetChildren())
            if not children:
                return None
            root_prim = children[0]

        if not root_prim.IsA(UsdGeom.Xformable):
            return None

        # 5. Compute the center of the bounding box
        bbox_min = box_range.GetMin()
        bbox_max = box_range.GetMax()
        center = Gf.Vec3d(
            (bbox_min[0] + bbox_max[0]) / 2.0,
            (bbox_min[1] + bbox_max[1]) / 2.0,
            (bbox_min[2] + bbox_max[2]) / 2.0,
        )

        # 6. Determine rotation to align longest axis to Z
        longest_axis = dims.index(max_dim)
        if align_z and longest_axis == 0:
            # X is longest → rotate 90° around Y axis (X→Z)
            rotation = Gf.Rotation(Gf.Vec3d(0, 1, 0), 90)
        elif align_z and longest_axis == 1:
            # Y is longest → rotate -90° around X axis (Y→Z)
            rotation = Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)
        else:
            # Z is already longest, or no alignment requested
            rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), 0)  # identity

        # 7. Compute scale factor
        target_size_meters = random.uniform(0.05, 0.25)
        scale_factor = target_size_meters / max_dim

        # 8. Compute translate to center at origin
        # xformOp order is [translate, orient, scale], applied right-to-left:
        #   P' = translate + R * (S * P)
        # For the bbox center C to map to origin:
        #   0 = translate + R * (S * C)
        #   translate = -R * (S * C)
        scaled_center = Gf.Vec3d(
            center[0] * scale_factor,
            center[1] * scale_factor,
            center[2] * scale_factor,
        )

        # Apply rotation to the scaled center
        rot_matrix = Gf.Matrix4d(1.0)  # identity 4x4
        rot_matrix.SetRotateOnly(rotation)
        rotated_scaled_center = rot_matrix.TransformDir(scaled_center)

        translate = Gf.Vec3d(
            -rotated_scaled_center[0],
            -rotated_scaled_center[1],
            -rotated_scaled_center[2],
        )

        # 9. Convert rotation to quaternion
        rot_quat = rotation.GetQuat()
        orient = Gf.Quatd(
            rot_quat.GetReal(),
            rot_quat.GetImaginary()[0],
            rot_quat.GetImaginary()[1],
            rot_quat.GetImaginary()[2],
        )

        # 10. Clear root prim xformOps and rebuild with composite transform
        xform = UsdGeom.Xformable(root_prim)
        xform.ClearXformOpOrder()

        t_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        t_op.Set(translate)

        o_op = xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        o_op.Set(orient)

        s_op = xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
        s_op.Set(Gf.Vec3d(scale_factor, scale_factor, scale_factor))

        print(f"  Normalized: center=({translate[0]:.4f},{translate[1]:.4f},{translate[2]:.4f}), "
              f"longest_axis={'XYZ'[longest_axis]}→Z, scale={scale_factor:.4f}")

        return scale_factor

    def count_total_faces(self, stage):
        """Count total face count across all meshes in the stage."""
        total = 0
        for prim in Usd.PrimRange(stage.GetPseudoRoot()):
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                counts = mesh.GetFaceVertexCountsAttr().Get()
                if counts:
                    total += len(counts)
        return total

    def generate_box_uvs(self, root_prim, scale=10.0):
        """Generates Tri-Planar Box UVs for materials."""
        for prim in Usd.PrimRange(root_prim):
            if not prim.IsA(UsdGeom.Mesh):
                continue

            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
            face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()

            if not points or not face_vertex_indices:
                continue

            pv_api = UsdGeom.PrimvarsAPI(prim)
            if pv_api.HasPrimvar("st"):
                continue

            uvs = []
            idx_pointer = 0
            for count in face_vertex_counts:
                face_indices = face_vertex_indices[idx_pointer: idx_pointer + count]
                idx_pointer += count

                p0 = Gf.Vec3f(points[face_indices[0]])
                p1 = Gf.Vec3f(points[face_indices[1]])
                p2 = Gf.Vec3f(points[face_indices[2]])
                v1, v2 = p1 - p0, p2 - p0
                normal = Gf.Cross(v1, v2).GetNormalized()

                if abs(normal[0]) >= abs(normal[1]) and abs(normal[0]) >= abs(normal[2]):
                    u_idx, v_idx = 1, 2
                elif abs(normal[1]) >= abs(normal[0]) and abs(normal[1]) >= abs(normal[2]):
                    u_idx, v_idx = 0, 2
                else:
                    u_idx, v_idx = 0, 1

                for v_idx_in_face in face_indices:
                    p = points[v_idx_in_face]
                    uvs.append(Gf.Vec2f(p[u_idx] * scale, p[v_idx] * scale))

            pv = pv_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
            pv.Set(uvs)

    # ─────────────────────────────────────────────────────────────────────────
    # Manual verification
    # ─────────────────────────────────────────────────────────────────────────

    def extract_mesh_data(self, stage):
        """Extract all mesh vertices and triangle faces from a USD stage for visualization."""
        all_vertices = []
        all_faces = []
        vertex_offset = 0

        for prim in Usd.PrimRange(stage.GetPseudoRoot()):
            if not prim.IsA(UsdGeom.Mesh):
                continue

            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
            face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()

            if not points or not face_vertex_indices:
                continue

            # Get the world transform for this mesh
            xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            world_xform = xform_cache.GetLocalToWorldTransform(prim)

            # Transform points to world space
            for pt in points:
                p = Gf.Vec3d(pt[0], pt[1], pt[2])
                p_world = world_xform.Transform(p)
                all_vertices.append([p_world[0], p_world[1], p_world[2]])

            # Triangulate faces and offset indices
            idx = 0
            for count in face_vertex_counts:
                if count >= 3:
                    # Fan triangulation
                    for tri in range(count - 2):
                        all_faces.append([
                            face_vertex_indices[idx] + vertex_offset,
                            face_vertex_indices[idx + tri + 1] + vertex_offset,
                            face_vertex_indices[idx + tri + 2] + vertex_offset,
                        ])
                idx += count

            vertex_offset += len(points)

        if not all_vertices:
            return None, None

        return np.array(all_vertices), np.array(all_faces)

    def show_mesh_for_approval(self, stage, uid, annotation_name=""):
        """
        Displays a 3D visualization of the mesh and asks the user for approval.
        Returns True if accepted, False if rejected.
        """
        vertices, faces = self.extract_mesh_data(stage)

        if vertices is None:
            print(f"  [!] No mesh data found for {uid}. Rejecting.")
            return False

        title = f"Asset: {annotation_name or uid}"

        if HAS_MATPLOTLIB and len(faces) > 0:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Subsample faces if too many (for rendering speed)
            max_render_faces = 5000
            if len(faces) > max_render_faces:
                indices = np.random.choice(len(faces), max_render_faces, replace=False)
                render_faces = faces[indices]
            else:
                render_faces = faces

            # Build polygon collection
            polygons = vertices[render_faces]
            mesh_collection = Poly3DCollection(polygons, alpha=0.6, linewidths=0.1, edgecolors='gray')
            mesh_collection.set_facecolor([0.4, 0.6, 0.8, 0.6])
            ax.add_collection3d(mesh_collection)

            # Auto-scale axes
            mins = vertices.min(axis=0)
            maxs = vertices.max(axis=0)
            center = (mins + maxs) / 2
            extent = (maxs - mins).max() / 2 * 1.2

            ax.set_xlim(center[0] - extent, center[0] + extent)
            ax.set_ylim(center[1] - extent, center[1] + extent)
            ax.set_zlim(center[2] - extent, center[2] + extent)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f"{title}\nVertices: {len(vertices)}, Faces: {len(faces)}\nClose window then answer Y/N in terminal")

            plt.tight_layout()
            plt.show(block=True)
        else:
            # Fallback: text-only
            print(f"\n  ┌─ {title}")
            print(f"  │  Vertices: {len(vertices)}")
            print(f"  │  Faces: {len(faces) if faces is not None else 0}")
            if vertices is not None:
                dims = vertices.max(axis=0) - vertices.min(axis=0)
                print(f"  │  Dimensions (m): {dims[0]:.3f} x {dims[1]:.3f} x {dims[2]:.3f}")
            print(f"  └─")

        # Ask for approval
        while True:
            response = input(f"  Accept '{annotation_name or uid}'? [Y/n]: ").strip().lower()
            if response in ('y', 'yes', ''):
                return True
            elif response in ('n', 'no'):
                return False
            else:
                print("  Please enter Y or N.")

    # ─────────────────────────────────────────────────────────────────────────
    # Main pipeline
    # ─────────────────────────────────────────────────────────────────────────

    async def run_pipeline(self, num_assets):
        # 1. Download (with pre-filtering)
        downloaded_assets = self.download_objaverse_assets(num_assets)
        annotations = objaverse.load_annotations()
        total = len(downloaded_assets)

        for i, (uid, source_filepath) in enumerate(downloaded_assets.items()):
            ann = annotations.get(uid, {})
            ann_name = ann.get('name', uid)
            print(f"\n{'='*60}")
            print(f"[{i+1}/{total}] Processing: {ann_name}")
            print(f"  UID: {uid}")
            print(f"  Source: {source_filepath}")

            target_usd_path = os.path.abspath(os.path.join(self.output_dir, f"distractor_{uid}.usd"))

            # 2. Convert to USD
            success = await self.convert_to_usd(source_filepath, target_usd_path)
            if not success:
                print(f"  [SKIP] Conversion failed.")
                self.stats["rejected_auto"] += 1
                continue

            # 3. Open stage for post-processing
            stage = Usd.Stage.Open(target_usd_path)
            if not stage:
                print(f"  [SKIP] Failed to open USD stage.")
                self.stats["rejected_auto"] += 1
                continue

            # 4. Count faces (post-conversion verification)
            face_count = self.count_total_faces(stage)
            print(f"  Faces: {face_count}")
            if face_count > MAX_FACE_COUNT:
                print(f"  [SKIP] Too many faces ({face_count} > {MAX_FACE_COUNT})")
                self.stats["rejected_auto"] += 1
                os.remove(target_usd_path)
                continue
            if face_count < MIN_FACE_COUNT:
                print(f"  [SKIP] Too few faces ({face_count} < {MIN_FACE_COUNT})")
                self.stats["rejected_auto"] += 1
                os.remove(target_usd_path)
                continue

            self.stats["converted"] += 1

            # 5. Post-process: strip physics
            self.strip_physics_properties(stage)

            # 6. Post-process: fix xformOp precision to double (MUST be before normalize)
            self.fix_xform_ops_to_double_precision(stage)

            # 7. Post-process: normalize geometry (center at origin, align Z, scale)
            scale_factor = self.normalize_geometry(stage, align_z=self.align_z)
            if scale_factor is None:
                print(f"  [SKIP] Degenerate geometry (zero-size bounding box)")
                self.stats["rejected_auto"] += 1
                os.remove(target_usd_path)
                continue

            # 8. Generate box UVs
            root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
            self.generate_box_uvs(root)

            # 9. Save the processed USD so the viewer sees the final result
            stage.GetRootLayer().Save()

            # 10. MANUAL VERIFICATION — show to user and ask Y/N
            print(f"  Opening visualization for manual review...")
            accepted = self.show_mesh_for_approval(stage, uid, ann_name)

            if not accepted:
                print(f"  [REJECTED] by user.")
                self.stats["rejected_manual"] += 1
                os.remove(target_usd_path)
                # Also clean up any sidecar files the converter may have created
                sidecar_dir = target_usd_path.replace(".usd", "")
                if os.path.isdir(sidecar_dir):
                    import shutil
                    shutil.rmtree(sidecar_dir, ignore_errors=True)
                continue

            # 11. ACCEPTED — register in config
            print(f"  [ACCEPTED] ✓")
            self.stats["accepted"] += 1

            part_key = f"distractor_{uid}"
            self.objects_config["parts"][part_key] = {
                "name": part_key,
                "usd_filepath": target_usd_path,
                "class": "1",
                "scale_factor": [scale_factor, scale_factor, scale_factor]
            }

            # Save config after each accepted asset (don't lose progress if interrupted)
            with open(self.config_path, 'w') as f:
                json.dump(self.objects_config, f, indent=4)

        # Final summary
        print(f"\n{'='*60}")
        print(f"Pipeline Complete!")
        print(f"  Downloaded:       {self.stats['downloaded']}")
        print(f"  Converted:        {self.stats['converted']}")
        print(f"  Auto-rejected:    {self.stats['rejected_auto']}")
        print(f"  Manual-rejected:  {self.stats['rejected_manual']}")
        print(f"  ACCEPTED:         {self.stats['accepted']}")
        print(f"  Config saved to:  {self.config_path}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Ingest graspable objects from Objaverse for bin-picking SDG")
    parser.add_argument('--num_assets', type=int, default=5000,
                        help='Number of objects to download and evaluate (after auto-filtering)')
    parser.add_argument('--config', type=str, default="./Config/objects.json",
                        help='Path to objects config JSON')
    parser.add_argument('--output', type=str, default="./assets/distractors/",
                        help='Output directory for accepted USD files')
    parser.add_argument('--no-align-z', action='store_true', default=False,
                        help='Disable automatic alignment of longest axis to Z')
    args = parser.parse_args()

    pipeline = MassIngestionPipeline(
        config_path=args.config,
        output_dir=args.output,
        align_z=not args.no_align_z,
    )
    asyncio.get_event_loop().run_until_complete(pipeline.run_pipeline(args.num_assets))


if __name__ == "__main__":
    main()
    simulation_app.close()