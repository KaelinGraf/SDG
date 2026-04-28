# Helper functions for scene_building that relate to Omniverse Replicator.
# This includes setting up/randomising sensors, defining materials, and defining annotators/writers

import numpy as np
from pathlib import Path

# Structured light imports (optional — only needed if structured_light=True)
try:
    import isaacsim.zivid as zivid_sim
    from isaacsim.zivid.camera import (
        SamplingMode,
        ZividCamera,
        spawn_zivid_casing,
        ZividCameraModelName,
        CameraManager,
    )
    from isaacsim.zivid import transforms as Ztransforms

    _HAS_ZIVID = True
except ImportError:
    _HAS_ZIVID = False

# Monocular imports (always available)
from camera_monocular import MonocularCameraManager, MonocularWriter, register_monocular_writer

import omni.replicator.core as rep
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.xforms import get_world_pose
import open3d as o3d
from omni.kit.viewport.utility import create_viewport_window
import omni.ui as ui

SETTINGS_YAML = "/home/kaelin/BinPicking/Pose_R_CNN/configs/zivid_config_specular.yml"


class RepCam:
    """
    Unified camera capture rig for synthetic data generation.

    Supports two capture modes:
      - **Structured Light** (structured_light=True): Uses ZividCameraImpl +
        CameraManager for phase-projection profilometry and point cloud
        reconstruction. Requires the isaacsim.zivid extension.
      - **Monocular RGB+Depth** (structured_light=False): Uses
        MonocularCameraManager for single path-traced RGB capture + perfect
        z-buffer. Much simpler and faster pipeline.

    Both modes output data in the same format (rgb.png, instance_raw.png,
    scene_info.json) for compatibility with the ReplicatorDataset dataloader.
    Monocular mode additionally outputs depth.npy.

    Usage:
        rep_cam = RepCam(
            bin_bounds=[...],
            bin_position=[...],
            output_dir="/path/to/output",
            structured_light=False,       # ← monocular mode
            resolution=(1280, 1280),       # ← only for monocular mode
        )
        rep_cam.init_cam()

        # Per scene:
        rep_cam.camera_manager.set_cam_world_pos_by_idx(
            idx, pos_world, rot_matrix, degrees=False, rot_format="matrix"
        )
        rep_cam.capture()
    """

    def __init__(
        self,
        bin_bounds=None,
        bin_position=None,
        focal_length=0.6,
        output_dir=None,
        structured_light=True,
        resolution=(1280, 1280),
        spp=128,
        env_path="/Env_",
    ):
        """
        Args:
            bin_bounds: [x_half, y_half, z_half, ...] bounding extents of the bin.
            bin_position: [x, y, z, ...] world position of the bin center.
            focal_length: Camera focal distance (used for initial height offset).
            output_dir: Root directory for saving captured data.
            structured_light: If True, use Zivid structured light pipeline.
                              If False, use monocular RGB + depth pipeline.
            resolution: (width, height) for monocular cameras (ignored for SL).
            spp: Samples per pixel for path tracing (monocular only).
            env_path: Environment prim path prefix for camera discovery.
        """
        if bin_bounds is None:
            bin_bounds = [0, 0, 0, 0, 0, 0]
        if bin_position is None:
            bin_position = [0, 0, 0, 0, 0, 0]

        self.bin_bounds = bin_bounds
        self.focal_length = focal_length
        self.output_dir = output_dir
        self.structured_light = structured_light
        self.resolution = resolution

        init_translation = [
            bin_position[0],
            bin_position[1],
            self.focal_length + bin_position[2] + bin_bounds[2],
            0, 0, 0,
        ]

        if structured_light:
            if not _HAS_ZIVID:
                raise ImportError(
                    "isaacsim.zivid extension not available. "
                    "Install it or use structured_light=False for monocular mode."
                )
            self.camera_manager = CameraManager(
                env_path=env_path,
                output_dir=output_dir,
            )
        else:
            # Register the monocular writer with Replicator
            register_monocular_writer()

            self.camera_manager = MonocularCameraManager(
                env_path=env_path,
                output_dir=output_dir,
                resolution=resolution,
                spp=spp,
            )

    def init_cam(self):
        """Initialize all cameras (create render products, attach annotators)."""
        self.camera_manager.initialize_cameras()

    def capture(self):
        """
        Capture a synchronized frame from all cameras.

        For structured light: runs the full phase projection sequence.
        For monocular: single path-traced render + z-buffer read.
        """
        self.camera_manager.capture_sequence()

    def view_frames(self):
        """Load and visualize captured frames from disk."""
        if self.structured_light:
            self.camera_manager.load_assets_from_disk()
        else:
            print(
                f"Monocular frames saved to: {self.camera_manager.output_dir}. "
                "Use the dataset viewer to inspect."
            )

    def set_cam_world_pos_by_idx(
        self, idx, translation, rotation, degrees=False, rot_format="matrix"
    ):
        """
        Set a camera's world pose by index.

        This is the primary interface for positioning cameras during data
        generation. Typically called with output from orbit_point():

            (pos, rot) = orbit_point(bin_pos, dist, elev, azim, look_at=True)
            rep_cam.set_cam_world_pos_by_idx(idx, pos, rot, rot_format="matrix")

        Args:
            idx: Camera index.
            translation: [x, y, z] world position.
            rotation: Rotation (format determined by rot_format).
            degrees: Whether Euler angles are in degrees (ignored for matrix).
            rot_format: "euler", "quat", or "matrix".
        """
        self.camera_manager.set_cam_world_pos_by_idx(
            idx, translation, rotation, degrees=degrees, rot_format=rot_format
        )

    def set_pose(self, trans, rot, quat=False, local=False):
        """
        Legacy pose setter for the Zivid camera rig.
        Only available in structured light mode.

        Args:
            trans: [x, y, z] translation.
            rot: [rx, ry, rz] euler or [w, x, y, z] quaternion.
            quat: If True, rot is quaternion format.
            local: If True, translation is relative to parent prim.
        """
        if not self.structured_light:
            raise NotImplementedError(
                "set_pose() uses the Zivid transform API. "
                "Use set_cam_world_pos_by_idx() for monocular mode."
            )

        transform = self.zivid_camera.get_local_pose()

        if not quat and rot is not None:
            zivid_rot = Ztransforms.Rotation.from_euler(euler=rot, degrees=True)
            transform.rot = zivid_rot
        elif quat and rot is not None:
            zivid_rot = Ztransforms.Rotation.from_quat(quat=rot)
            transform.rot = zivid_rot

        if trans is not None:
            transform.t = trans

        self.zivid_camera.set_local_pose(transform)