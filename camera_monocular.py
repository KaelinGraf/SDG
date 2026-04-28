# Kaelin Graf-Ogilvie 2025
# Monocular RGB + Depth camera for Isaac Sim synthetic data generation.
# Replaces structured light pipeline with single path-traced capture + z-buffer.
# Outputs data compatible with the existing ReplicatorDataset dataloader.

from __future__ import annotations
import numpy as np
import cv2
import omni.replicator.core as rep
import omni.timeline
import omni.usd
import carb.settings
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import os
import json
import re
import isaacsim.core.utils.numpy.rotations as rot_utils

from pxr import UsdGeom, Gf, Usd, Sdf
from omni.replicator.core import WriterRegistry, AnnotatorRegistry, BackendDispatch, Writer
from omni.isaac.core.utils import prims
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils import prims as prim_utils
from isaacsim.sensors.camera import USD_CAMERA_TENTHS_TO_STAGE_UNIT, Camera

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

Y_DOWN_Z_FORWARD = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
# ---------------------------------------------------------------------------
# Monocular Depth Camera
# ---------------------------------------------------------------------------

class MonocularDepthCamera:
    """
    A single monocular camera in Isaac Sim that captures:
      - Path-traced RGB image
      - Perfect z-buffer (depth)
      - Instance segmentation mask
      - Scene metadata (poses, occlusion, camera params)

    Designed as a drop-in replacement for ZividCameraImpl, removing all
    structured light complexity.
    """

    def __init__(
        self,
        prim_path: str,
        resolution: Tuple[int, int] = (1280, 1280),
        output_dir: Optional[str] = None,
        annotators: Optional[List[str]] = None,
    ):
        """
        Args:
            prim_path: USD path where the camera prim will be created/found.
            resolution: (width, height) of the rendered image.
            output_dir: Root directory for saving captures.
            annotators: Unused — kept for API parity. Annotators are owned
                        by the MonocularWriter (same pattern as ZividWriter).
        """
        self.prim_path = prim_path
        self.resolution = resolution
        self.output_dir = output_dir
        self._render_product = None
        self._initialized = False
        self._prim = SingleXFormPrim(
            prim_path,
            name="rgbd_sensor",
        )
        self._camera_sensor_prim_path = f"{prim_path}/rgbd_sensor"
        
        self.camera = self._create_camera_sensor(self._camera_sensor_prim_path, "rgbd_sensor")

    def _create_camera_sensor(self, prim_path: str, name: str) -> Camera:
        # This is a hacky way to avoid warnings because that the camera  init methods sets a resolution
        # that does not match the default aspect ratio of the camera in USD
        prim = prim_utils.get_prim_at_path(prim_path) if prim_utils.is_prim_path_valid(prim_path) else None
        if prim and prim.GetTypeName() == "Camera":
            # reset to default aspect ratio if camera already exists
            prim.GetAttribute("verticalAperture").Set(24.0)
            prim.GetAttribute("horizontalAperture").Set(24.0)

        camera = Camera(
            prim_path=prim_path,
            name=name,
            #translation = Gf.Vec3d(0, 0, 0),
            orientation =  np.array([0.0,0.707,0.0,0.707]),
            resolution=self.resolution,
        )

        return camera
    # ---- Setup / lifecycle ------------------------------------------------

    def initialize(self):
        """Create camera prim and render product."""
        if self._initialized:
            return
        self.camera.initialize()
        self._render_product_path = self.camera.get_render_product_path()

        self._initialized = True

    def set_render_product_updates(self, enabled: bool):
        """Enable/disable render product updates (for perf during non-capture steps)."""
        if enabled:
            self.camera.resume()
        else:
            self.camera.pause()

    # ---- Pose control -----------------------------------------------------

    def set_world_pose(self, translation: np.ndarray, rotation_matrix: np.ndarray):
        """
        Set camera world pose.

        Args:
            translation: [3] world position in meters.
            rotation_matrix: [3,3] rotation matrix (camera-to-world).
        """
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self.prim_path)
        xformable = UsdGeom.Xformable(prim)

        # Clear existing ops and set as a single 4x4 transform
        xformable.ClearXformOpOrder()
        tf = Gf.Matrix4d()
        tf.SetRow(0, Gf.Vec4d(*rotation_matrix[0], 0))
        tf.SetRow(1, Gf.Vec4d(*rotation_matrix[1], 0))
        tf.SetRow(2, Gf.Vec4d(*rotation_matrix[2], 0))
        tf.SetRow(3, Gf.Vec4d(*translation, 1))
        xformable.AddTransformOp().Set(tf)

    def set_local_pose(self, translation: np.ndarray, rotation_matrix: np.ndarray):
        """Set camera local pose (relative to parent prim)."""
        self.set_world_pose(translation, rotation_matrix)
# ---------------------------------------------------------------------------
# Monocular Camera Manager
# ---------------------------------------------------------------------------

class MonocularCameraManager:
    """
    Manages the lifecycle and synchronized capture of multiple MonocularDepthCamera
    instances across Isaac Sim environments.

    Drop-in replacement for CameraManager, removing all structured light logic.
    Uses the same writer attachment + schedule_write pattern as CameraManager.
    """

    def __init__(
        self,
        env_path: str,
        output_dir: str,
        resolution: Tuple[int, int] = (1280, 1280),
        spp: int = 128,
        annotators: Optional[List[str]] = None,
    ):
        """
        Args:
            env_path: Prefix path to environments (e.g. "/Env").
                      Wildcard matching finds all "/Env*/Camera" containers.
            output_dir: Root output directory. A batch subdirectory is created.
            resolution: (width, height) for all cameras.
            spp: Samples per pixel for path tracing.
            annotators: Annotator list passed to each camera.
        """
        self.output_dir = output_dir
        self.resolution = resolution
        self.spp = spp
        self.kit = omni.kit.app.get_app()
        self.timeline = omni.timeline.get_timeline_interface()

        self._create_new_batch_directory()

        # Discover environment containers
        container_paths = prims.find_matching_prim_paths(f"{env_path}*")
        print(f"Found {len(container_paths)} environments matching '{env_path}*'")

        # Create a camera per environment, each with its own writer
        self.cameras: List[MonocularDepthCamera] = []
        self.writers: List[MonocularWriter] = []
        for path in container_paths:
            camera = MonocularDepthCamera(
                prim_path=f"{path}/MonocularCamera",
                resolution=resolution,
                output_dir=self.output_dir,
                annotators=annotators,
            )
            self.cameras.append(camera)

            # Each camera gets its own writer (mirrors ZividWriter per camera)
            writer = MonocularWriter(output_dir=self.output_dir)
            self.writers.append(writer)
            print(f"Created monocular camera at {path}/MonocularCamera")

        self.num_cams = len(self.cameras)
        print(f"Total cameras: {self.num_cams}")

    def _create_new_batch_directory(self):
        """Create a new batch directory for this capture session."""
        if self.output_dir is None:
            raise ValueError("output_dir cannot be None")

        # Auto-increment batch number
        base = Path(self.output_dir)
        existing = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("batch_")] if base.exists() else []
        batch_num = len(existing) + 1
        self.output_dir = str(base / f"batch_{batch_num}")
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory: {self.output_dir}")

    def initialize_cameras(self):
        """Initialize all cameras and attach writers to their render products."""
        self.carb_setup()  # Set global render settings for path tracing
        for camera, writer in zip(self.cameras, self.writers):
            camera.initialize()
            camera.set_render_product_updates(enabled=False)
            rp_node = rep.create.render_product(camera._camera_sensor_prim_path, camera.resolution)
            # Attach writer to this camera's render product (no trigger — manual schedule)
            writer.attach([rp_node], trigger=None)

    def capture_sequence(self):
        """
        Capture a synchronized frame from all cameras.

        Follows the same flow as CameraManager.capture_sequence_sl:
          1. Enable render products
          2. Pause timeline, step replicator with path tracing SPP
          3. Set frame directories on each writer via update_dir()
          4. Schedule writes + orchestrator.step() to trigger write()
          5. Disable render products, resume timeline
        """
        settings = carb.settings.get_settings()

        # Enable render products for capture
        for camera in self.cameras:
            camera.set_render_product_updates(enabled=True)

        # Pause physics, render with accumulation
        self.timeline.pause()
        self._step_replicator(n=self.spp)

        # Set frame directories on each writer
        for writer in self.writers:
            frame_count = len([f for f in os.listdir(self.output_dir) if f.startswith("frame_")])
            frame_dir = os.path.join(self.output_dir, f"frame_{frame_count}")
            os.makedirs(frame_dir, exist_ok=True)
            writer.update_dir(frame_dir)

        # Trigger writes via Replicator orchestrator (calls writer.write(data))
        self.schedule_writes(rp_enabled=True)

        # Disable render products
        for camera in self.cameras:
            camera.set_render_product_updates(enabled=False)

        # Reset to low SPP for interactive performance
        settings.set("/rtx/pathtracing/spp", 1)
        settings.set("/rtx/pathtracing/totalSpp", 1)
        self.timeline.play()

    def schedule_writes(self, rp_enabled: bool = True):
        """
        Schedule writes on all writers, then step the orchestrator.
        Mirrors CameraManager.schedule_writes().
        """
        for camera, writer in zip(self.cameras, self.writers):
            if not rp_enabled:
                camera.set_render_product_updates(enabled=True)
            writer.schedule_write()

        rep.orchestrator.step(rt_subframes=0, pause_timeline=True)

        if not rp_enabled:
            for camera in self.cameras:
                camera.set_render_product_updates(enabled=False)
                
    def carb_setup(self):
        settings = carb.settings.get_settings()
        #settings.set("/omni/replicator/captureOnPlay", False)
        #settings.set("/rtx/rendermode", "realTime")
        settings.set("/rtx/rendermode", "PathTracing")
        settings.set("/rtx/pathtracing/maxBounces", 3) 
        settings.set("/rtx/pathtracing/adaptiveSampling/enabled", True)
        settings.set("/rtx/resetPtAccumOnAnimTimeChange", True)
        settings.set("/rtx/pathtracing/spp", 1) 
        settings.set("/rtx/pathtracing/totalSpp", 1)
        settings.set("/rtx/pathtracing/adaptiveSampling/targetError",0.001)
        settings.set("/rtx/pathtracing/optixDenoiser/enabled", False)
        #disaple path tracing aa
        settings.set("/rtx/pathtracing/aa/op", 1)
        settings.set("/rtx/pathtracing/aa/filterRadius", 0.0)
        settings.set("/rtx/pathtracing/cached/enabled", False)
        settings.set("/rtx/post/dlss/execMode", 2)
        settings.set_bool("/rtx/directLighting/sampledLighting/enabled", False)
        settings.set_bool("/rtx/ambientOcclusion/enabled",True)
        settings.set_bool("/rtx/indirectDiffuse/enabled",True)
        settings.set_int("/rtx/indirectDiffuse/maxIndirectDiffuseBounces", 2)
        settings.set_bool("/rtx/post/bloom/enabled", False)
        settings.set_bool("/rtx/post/lensFlares/enabled", False)

        settings.set_int("/rtx/texturestreaming/mode", 0) # Force full-res textures
        settings.set_int("/rtx/post/aa/op", 0)            # Disable TAA (prevents ghosting/blur)
        settings.set_bool("/rtx/indirectLighting/enabled", True)
        settings.set_bool("/rtx/post/autoExposure/enabled", False)

        settings.set_int("/rtx/post/tonemap/op", 1) # 1 = Linear
        #settings.set_float("/rtx/post/tonemap/whitepoint", 1.0) # Default whitepoint
        settings.set_bool("/rtx/materialDb/syncLoads", True)
        settings.set_int("/rtx/material/textureFilterMode", 0)
        settings.set_bool("/rtx/hydra/materialSyncLoads", True)
        settings.set_int("/rtx/material/textureFilterMode", 0) # 0 = Linear, 1 = Cubic?
    
        return settings

    def _step_replicator(self, n: int = 128):
        """
        Step the renderer with path tracing accumulation.

        Args:
            n: Total samples per pixel to accumulate.
        """
        settings = carb.settings.get_settings()
        spp_per_update = 8
        settings.set("/rtx/pathtracing/spp", spp_per_update)
        settings.set("/rtx/pathtracing/totalSpp", n)

        updates = int(np.ceil(n / spp_per_update)) + 2
        for _ in range(updates):
            self.kit.update()

    def set_cam_world_pos_by_idx(
        self,
        idx: int,
        translation: np.ndarray,
        rotation: np.ndarray,
        degrees: bool = True,
        rot_format: str = "euler",
    ):
        """
        Set a camera's world pose by index.

        Args:
            idx: Camera index.
            translation: [x, y, z] position in meters.
            rotation: Rotation in the format specified by rot_format.
            degrees: Whether Euler angles are in degrees.
            rot_format: "euler", "quat", or "matrix".
        """
        if idx >= len(self.cameras):
            raise IndexError(f"Camera index {idx} out of bounds (have {len(self.cameras)})")

        from scipy.spatial.transform import Rotation as R

        if rot_format == "euler":
            rot_matrix = R.from_euler("xyz", np.asarray(rotation), degrees=degrees).as_matrix()
        elif rot_format == "quat":
            rot_matrix = R.from_quat(np.asarray(rotation)).as_matrix()
        elif rot_format == "matrix":
            rot_matrix = np.asarray(rotation)
        else:
            raise ValueError(f"Unknown rot_format: {rot_format}")

        self.cameras[idx].set_world_pose(np.asarray(translation), rot_matrix)


# ---------------------------------------------------------------------------
# Monocular Writer
# ---------------------------------------------------------------------------

class MonocularWriter(Writer):
    """
    Replicator Writer for monocular RGB+Depth captures.

    Follows the same architecture as ZividWriter:
      - Registers annotators in __init__ (so the base Writer class attaches them
        to the render product when writer.attach() is called).
      - Overrides write(self, data: dict) which is called by Replicator's
        orchestrator when schedule_write() triggers.
      - Frame output directory is updated externally via update_dir() before
        each schedule_write() call.

    Output per frame:
      - rgb.png              : Path-traced RGB image (BGR, uint8)
      - depth.npy            : Z-buffer depth map (float32, meters)
      - *_instance_raw.png   : 16-bit instance segmentation mask
      - *_instance_output.png: Colorized instance visualization
      - *_scene_info.json    : Scene metadata (camera params, per-object
                               class, segmentation_id, visibility_ratio, pose)
    """

    def __init__(self, output_dir: str, annotators: Optional[List[str]] = None):
        self.annotators = []
        self.data_structure = "renderProduct"
        self.backend = BackendDispatch(output_dir=output_dir)
        self.output_dir = output_dir
        self.frame_dir = None

        if annotators is None:
            annotators = [
                "LdrColor",  # RGB image
                "distance_to_image_plane",
                "instance_segmentation",
                "CameraParams",
            ]

        for annotator_name in annotators:
            self.annotators.append(AnnotatorRegistry.get_annotator(annotator_name))

    def update_dir(self, frame_dir: str):
        """Set the output directory for the next write() call."""
        self.frame_dir = frame_dir

    def get_dir(self) -> Optional[str]:
        """Return current frame output directory."""
        return self.frame_dir

    def write(self, data: dict):
        """
        Called by Replicator orchestrator after schedule_write().

        Args:
            data: Standard Replicator data dict with structure:
                  {"renderProducts": {rp_name: {annotator_name: annotator_data, ...}}}
        """
        for rp_name, annotator_data in data["renderProducts"].items():
            #print(annotator_data)
            if self.frame_dir is None:
                print("Warning: frame_dir not set, skipping write")
                return

            os.makedirs(self.frame_dir, exist_ok=True)

            # --- RGB image ---
            rgb_data = annotator_data.get("LdrColor")
            if rgb_data is not None:
                #rgb data is a dict with {'data': array} where array is (H, W, 4) RGBA in uint8. We convert to BGR and drop alpha for saving.
                rgb_data = rgb_data['data'].astype(np.uint8)
                rgb_data = rgb_data[:, :, :3][..., ::-1]
                cv2.imwrite(os.path.join(self.frame_dir, "rgb.png"), rgb_data)

            # --- Depth buffer ---
            depth_data = annotator_data.get("distance_to_image_plane")
            if depth_data is not None:
                depth_arr = np.asarray(depth_data['data'], dtype=np.float32)
                np.save(os.path.join(self.frame_dir, "depth.npy"), depth_arr)

            # --- Instance segmentation mask + scene info ---
            seg_data = annotator_data.get("instance_segmentation")
            cam_params = annotator_data.get("CameraParams")

            if seg_data is None or cam_params is None:
                print(f"Warning: Missing segmentation or camera data for {self.frame_dir}")
                return

            # Save raw instance mask as 16-bit PNG + colorized visualization
            segment_id_pairs = _save_instance_mask(seg_data, self.frame_dir, rp_name)

            # Build camera intrinsics
            w_res = cam_params["renderProductResolution"][0]
            h_res = cam_params["renderProductResolution"][1]
            pixel_size = cam_params["cameraAperture"][0] / w_res
            fx = float(cam_params["cameraFocalLength"] / pixel_size)
            fy = float(cam_params["cameraFocalLength"] / pixel_size)
            cx = float(w_res / 2.0 + cam_params["cameraApertureOffset"][0])
            cy = float(h_res / 2.0 + cam_params["cameraApertureOffset"][1])
            cam_K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]

            # Compute world-to-camera transform (bypass Replicator, use USD directly)
            import omni.timeline
            stage = omni.usd.get_context().get_stage()
            timeline = omni.timeline.get_timeline_interface()
            time_code = Usd.TimeCode(
                timeline.get_current_time() * stage.GetTimeCodesPerSecond()
            )

            cam_prim_path = annotator_data.get("camera", "")
            if not cam_prim_path:
                print("Warning: 'camera' not found in annotator_data. Skipping.")
                continue

            cam_prim = stage.GetPrimAtPath(cam_prim_path)
            cam_l2w = np.array(
                UsdGeom.Xformable(cam_prim).ComputeLocalToWorldTransform(time_code)
            )
            w2c_tf = np.linalg.inv(cam_l2w)

            # USD → OpenCV coordinate transform
            T_usd_to_cv = np.array([
                [1.0,  0.0,  0.0, 0.0],
                [0.0, -1.0,  0.0, 0.0],
                [0.0,  0.0, -1.0, 0.0],
                [0.0,  0.0,  0.0, 1.0],
            ])
            w2c_tf_cv = np.matmul(w2c_tf, T_usd_to_cv)

            R_w2c_pure = w2c_tf_cv[:3, :3]
            t_w2c_pure = w2c_tf_cv[3, :3]
            cam_R_w2c = R_w2c_pure.T.flatten().tolist()
            cam_t_w2c = t_w2c_pure.tolist()

            # --- Build per-object scene info ---
            objects_list = []

            for str_id, semantic_label in seg_data.get("idToSemantics", {}).items():
                id_int = int(str_id)
                prim_path = seg_data.get("idToLabels", {}).get(str_id, "")

                # Skip table/ground prims
                if "table_xform" in prim_path:
                    continue

                try:
                    seg_label = segment_id_pairs[id_int]
                except Exception:
                    seg_label = -1

                # Clean class name
                raw_class = str(semantic_label.get("class", "unknown"))
                clean_class = re.sub(r"_instance_\d+$", "", raw_class)

                # Get object prim for pose computation
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid() or not prim.IsA(UsdGeom.Xformable):
                    continue

                l2w_tf = np.array(
                    UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time_code)
                )

                # Shift origin to bounding box centroid (matching ZividWriter)
                bbox_cache = UsdGeom.BBoxCache(time_code, ["default", "render"])
                local_bound = bbox_cache.ComputeUntransformedBound(prim)

                if not local_bound.GetBox().IsEmpty():
                    centroid = local_bound.ComputeCentroid()
                    centroid_tf = np.eye(4)
                    centroid_tf[3, :3] = [centroid[0], centroid[1], centroid[2]]
                    l2w_tf = np.matmul(centroid_tf, l2w_tf)

                # Local-to-camera transforms
                loc2cam_tf_usd = np.matmul(l2w_tf, w2c_tf)
                loc2cam_tf = np.matmul(loc2cam_tf_usd, T_usd_to_cv)

                # --- Visibility ratio via triangle rasterization ---
                px_count_vis = int(np.count_nonzero(seg_data["data"] == id_int))
                if px_count_vis <= 0:
                    continue

                visibility_ratio = _compute_visibility_ratio(
                    prim, stage, time_code, w2c_tf, T_usd_to_cv,
                    cam_K, seg_data["data"], id_int, px_count_vis,
                    resolution=(int(w_res), int(h_res)),
                )

                # Extract pose (rotation + translation), removing scale
                R_row = loc2cam_tf[:3, :3]
                t_cam = loc2cam_tf[3, :3]

                scale_x = np.linalg.norm(R_row[0, :])
                scale_y = np.linalg.norm(R_row[1, :])
                scale_z = np.linalg.norm(R_row[2, :])
                scales = np.array([scale_x, scale_y, scale_z])
                scales[scales == 0] = 1.0

                R_pure_row = R_row / scales[:, np.newaxis]
                R_bop = R_pure_row.T  # BOP uses column-major

                pose = {
                    "cam_R_m2c": R_bop.flatten().tolist(),
                    "cam_t_m2c": t_cam.tolist(),
                    "scale_m2c": scales.tolist(),
                }

                objects_list.append({
                    "class": clean_class,
                    "prim_path": prim_path,
                    "segmentation_id": int(seg_label),
                    "pose": pose,
                    "visibility_ratio": visibility_ratio,
                    "occlusion_ratio": 1.0 - visibility_ratio,
                })

            # --- Write scene info JSON ---
            output = {
                "camera": {
                    "cam_K": cam_K,
                    "resolution": [int(w_res), int(h_res)],
                    "cam_R_w2c": cam_R_w2c,
                    "cam_t_w2c": cam_t_w2c,
                },
                "objects": objects_list,
            }

            json_path = os.path.join(self.frame_dir, f"{rp_name}_scene_info.json")
            with open(json_path, "w") as f:
                json.dump(output, f, indent=4)


# ---------------------------------------------------------------------------
# Module-level helper functions (used by MonocularWriter.write)
# ---------------------------------------------------------------------------

def _compute_visibility_ratio(
    prim: Usd.Prim,
    stage: Usd.Stage,
    time_code: Usd.TimeCode,
    w2c_tf: np.ndarray,
    T_usd_to_cv: np.ndarray,
    cam_K: List[float],
    seg_mask: np.ndarray,
    id_int: int,
    px_count_vis: int,
    resolution: Tuple[int, int] = (1280, 1280),
) -> float:
    """
    Compute accurate visibility ratio via mesh triangle rasterization.

    Instead of the convex hull (which inflates concave objects), this
    rasterizes every projected mesh triangle into a silhouette mask.
    """
    fx, fy, cx, cy = cam_K[0], cam_K[4], cam_K[2], cam_K[5]
    w, h = resolution

    silhouette = np.zeros((h, w), dtype=np.uint8)

    for mesh_prim in Usd.PrimRange(prim):
        if not mesh_prim.IsA(UsdGeom.Mesh):
            continue

        mesh = UsdGeom.Mesh(mesh_prim)
        pts = mesh.GetPointsAttr().Get()
        face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
        face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not face_vertex_indices:
            continue

        mesh_l2w = np.array(
            UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(time_code)
        )
        mesh_l2c = np.matmul(np.matmul(mesh_l2w, w2c_tf), T_usd_to_cv)

        pts_np = np.array(pts)
        hom_pts = np.hstack((pts_np, np.ones((pts_np.shape[0], 1))))
        cam_pts = np.matmul(hom_pts, mesh_l2c)[:, :3]

        z = cam_pts[:, 2]
        valid = z > 0.01
        uv = np.zeros((len(pts_np), 2), dtype=np.float32)
        uv[valid, 0] = (cam_pts[valid, 0] / z[valid]) * fx + cx
        uv[valid, 1] = (cam_pts[valid, 1] / z[valid]) * fy + cy

        idx_offset = 0
        for face_size in face_vertex_counts:
            face_indices = list(
                face_vertex_indices[idx_offset : idx_offset + int(face_size)]
            )
            idx_offset += int(face_size)

            if not all(valid[fi] for fi in face_indices):
                continue

            face_uvs = uv[face_indices].astype(np.int32)
            cv2.fillPoly(silhouette, [face_uvs], 255)

    full_area = int(np.count_nonzero(silhouette))
    if full_area <= 0:
        return 1.0

    return max(0.0, min(1.0, float(px_count_vis) / full_area))


def _save_instance_mask(
    seg_data: Dict, frame_dir: str, rp_name: str
) -> Dict[int, int]:
    """
    Save the instance segmentation mask as a 16-bit PNG and return
    the ID-to-index mapping. Mirrors plot_replicator_instance_mask from ZividWriter.
    """
    mask_32 = np.asarray(seg_data["data"], dtype=np.uint16)
    diff_labels = np.unique(mask_32)

    index_label_pairs = {}
    for index, label in enumerate(diff_labels):
        index_label_pairs[label] = index

    raw_save_path = os.path.join(frame_dir, f"{rp_name}_instance_raw.png")
    cv2.imwrite(raw_save_path, mask_32)

    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0.0, 1.0, len(diff_labels)))
    h, w = mask_32.shape
    rgb_canvas = np.zeros((h, w, 3))
    for idx, label in enumerate(diff_labels):
        rgb_canvas[mask_32 == label] = colors[idx][:3]

    plt.figure(figsize=(10, 6))
    plt.imshow(rgb_canvas)
    plt.axis("off")
    plt.savefig(
        os.path.join(frame_dir, f"{rp_name}_instance_output.png"),
        bbox_inches="tight",
    )
    plt.close()

    return index_label_pairs


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------

def register_monocular_writer():
    """Register MonocularWriter with Replicator's WriterRegistry."""
    WriterRegistry.register(MonocularWriter)
    if "MonocularWriter" not in WriterRegistry._default_writers:
        WriterRegistry._default_writers.append("MonocularWriter")