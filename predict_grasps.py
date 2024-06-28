"""
Implementation of grasp_synthesis.

Predicts 6DOF grasps on input point clouds.

Author: Tim Player, Marc Micatka

Note: torch.backends.cudnn.benchmark=True
makes a big difference on FPS for some PTS_PER_FRAME values,
but seems to increase memory usage and can result in OOM errors.
"""

# Standard Library
import os
import warnings
from typing import Optional, Tuple

import numpy as np
import torch

# Third-party
from omegaconf import OmegaConf

try:
    from .tsgrasp.net.lit_tsgraspnet import LitTSGraspNet
    from .tsgrasp_utils import (
        build_6dof_grasps,
        downsample_xyz,
        eul_to_rotm,
        generate_color_lookup,
        infer_grasps,
    )
except ImportError:
    from tsgrasp.net.lit_tsgraspnet import LitTSGraspNet  # type: ignore
    from tsgrasp_utils import (  # type: ignore
        build_6dof_grasps,
        downsample_xyz,
        eul_to_rotm,
        generate_color_lookup,
        infer_grasps,
    )

# Suppresses a userwarning from kornia
warnings.filterwarnings("ignore", category=UserWarning)


class GraspPredictor:
    """
    Node implementing tsgrasp network
    """

    def __init__(self, model_metadata: dict, verbose: bool, pkg_root: str) -> None:
        self.device = torch.device("cuda")

        if model_metadata is None:
            print("No Metadata Loaded!")

        # Default params, if not in metadata
        self.confidence_threshold = 0.0
        self.gripper_depth = 0.090
        self.offset_distance = 0.20
        self.downsample = 0.50
        self.queue_len = 1
        self.top_k = 500
        self.downsample_grasps = True
        self.outlier_threshold = 0.00005
        self.pts_per_frame = 45000
        self.verbose = verbose

        try:
            model_path = os.path.join(pkg_root, model_metadata["ckpt_path"])
            assert os.path.isfile(model_path)

            # Other model params from YAML:
            self.queue_len = model_metadata["queue_len"]
            self.outlier_threshold = model_metadata["outlier_threshold"]
            self.pts_per_frame = model_metadata["pts_per_frame"]
            model_cfg = OmegaConf.create(model_metadata["model"]["model_cfg"])
            training_cfg = OmegaConf.create(model_metadata["training"])

        except KeyError as ex:
            print(f"Key Error: {ex}")

        self.color_lookup = generate_color_lookup()

        self.pc_input_msg = None
        self.py_grasps = None
        self.tf_trans = [0, 0, 0]
        self.tf_rot = [0, 0, np.pi / 2]

        self.pl_model = LitTSGraspNet(model_cfg=model_cfg, training_cfg=training_cfg)
        self.pl_model.load_state_dict(torch.load(model_path)["state_dict"])

        # # load Pytorch Lightning network
        self.pl_model.to(self.device)
        self.pl_model.eval()

    @torch.inference_mode()
    def detect(self, pointcloud: np.array) -> Optional[Tuple]:
        """
        Run grasp prediction on a single input

        Args:
            pointcloud (np.array): input pointcloud

        Returns:
            Tuple(np.array, np.array): Tuple of grasps pointcloud and a heatmap pointcloud (np.array)
        """

        grasps_array: Optional[np.array] = None
        cm_array: Optional[np.array] = None

        try:
            orig_pts = [pointcloud]
            pts = [
                torch.from_numpy(pt.astype(np.float32)).to(self.device)
                for pt in orig_pts
            ]

        except ValueError as ex:
            print(f"Is this error because there are fewer than 300x300 points? - {ex}")
            return (None, None)

        pts = downsample_xyz(pts, self.pts_per_frame)
        if pts is None or any(len(pcl) == 0 for pcl in pts):
            if self.verbose:
                print("No points found after downsampling!")
            return (None, None)

        all_grasps, all_confs, all_widths = self.identify_grasps(pts)

        try:
            all_grasps = self.ensure_grasp_y_axis_upward(all_grasps)
            all_grasps = self.transform_to_eq_pose(all_grasps)

            (grasps_array, cm_array) = self.generate_pc_data(
                pts, all_grasps, all_confs, all_widths
            )

            return (grasps_array, cm_array)

        except RuntimeError as ex:
            print(f"Encountered Runtime Error! {ex}")
            return (None, None)

    def identify_grasps(self, pts):
        """
        Identify grasps in point cloud pts

        Args:
            pts (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        try:
            outputs = infer_grasps(
                self.pl_model, pts, grid_size=self.pl_model.model.grid_size
            )

            class_logits, baseline_dir, approach_dir, grasp_offset, positions = outputs

            grasps = build_6dof_grasps(
                positions,
                baseline_dir,
                approach_dir,
                grasp_offset,
                gripper_depth=self.gripper_depth,
            )

            confs = torch.sigmoid(class_logits)
            return grasps, confs, grasp_offset
        except Exception as ex:
            print(f"{ex}")
            return None, None, None

    def ensure_grasp_y_axis_upward(self, grasps: torch.Tensor) -> torch.Tensor:
        """
        Flip grasps with their Y-axis pointing downwards by 180 degrees about the wrist (z) axis,
            because we have mounted the camera on the wrist in the direction of the Y axis and don't
            want it to be scraped off on the table.

        Args:
            grasps (torch.Tensor): (N, 4, 4) grasp pose tensor

        Returns:
            torch.Tensor: (N, 4, 4) grasp pose tensor with some grasps flipped
        """

        ngrasps = len(grasps)

        # The strategy here is to create a  Boolean tensor for whether
        # to flip the grasp. From the way we mounted our camera, we know that
        # we'd prefer grasps with X axes that point up in the camera frame
        # (along the -Y axis). Therefore, we flip the rotation matrices of the
        # grasp poses that don't do that.

        # For speed, the flipping is done by allocating two (N, 4, 4) transformation
        # matrices: one for flipping (flips) and one for do-nothing (eyes). We select
        # between them with torch.where and perform matrix multiplication. This avoids
        # a for loop (~100X speedup) at the expense of a bit of memory and obfuscation.

        y_axis = torch.tensor([0, 1, 0], dtype=torch.float32).to(self.device)
        flip_about_z = torch.tensor(
            [[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float32
        ).to(self.device)

        needs_flipping = grasps[:, :3, 1] @ y_axis > 0
        needs_flipping = needs_flipping.reshape(ngrasps, 1, 1).expand(ngrasps, 3, 3)

        eyes = torch.eye(3).repeat((ngrasps, 1, 1)).to(self.device)
        flips = flip_about_z.repeat((ngrasps, 1, 1)).to(self.device)

        tfs = torch.where(needs_flipping, flips, eyes)

        grasps[:, :3, :3] = torch.bmm(grasps[:, :3, :3], tfs)
        return grasps

    def transform_to_eq_pose(self, poses):
        """
        Apply the static frame transformation between the network output and the
        input expected by the servoing logic at /panda/cartesian_impendance_controller/equilibrium_pose.

        The servoing pose is at the gripper pads, and is rotated, while the network output is at the wrist.

        This is an *intrinsic* pose transformation, where each grasp pose moves a fixed amount relative to
        its initial pose, so we right-multiply instead of left-multiply.
        """

        roll, pitch, yaw = self.tf_rot
        x, y, z = self.tf_trans
        tf = torch.cat(
            [
                torch.cat(
                    [
                        eul_to_rotm(roll, pitch, yaw),
                        torch.Tensor([x, y, z]).reshape(3, 1),
                    ],
                    dim=1,
                ),
                torch.Tensor([0, 0, 0, 1]).reshape(1, 4),
            ],
            dim=0,
        ).to(poses.device)
        return poses @ tf

    def generate_pc_data(
        self, pts, all_grasps, all_confs, all_widths
    ) -> Tuple[np.array, np.array]:
        """
        Returns point cloud of the grasps with confidences colormapped
        Also includes pc with all the grasps and widths.

        Args:
            pts (torch.Tensor): x, y, z points
            all_grasps (torch.Tensor): 4x4 pose matrix for each grasp
            all_confs (torch.Tensor): Confidence float values for each grasp
            all_widths (torch.Tensor): Width float values for each grasp

        Returns:
            Tuple[np.array, np.array]: pointcloud containing grasp information, colormapped pc
        """
        cloud_points = pts[-1]

        confs = all_confs.cpu().numpy()
        widths = all_widths.cpu().numpy()
        points = cloud_points.cpu().numpy()
        npoints = len(points)

        int_confs = np.round(confs * 255).astype(np.uint8).squeeze()
        colors = self.color_lookup[int_confs]

        cm_array = np.zeros(
            (npoints,),
            dtype=[
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("r", np.float32),
                ("g", np.float32),
                ("b", np.float32),
                ("a", np.float32),
            ],
        )
        cm_array["x"] = points[:, 0]
        cm_array["y"] = points[:, 1]
        cm_array["z"] = points[:, 2]
        cm_array["r"] = colors[:, 0]
        cm_array["g"] = colors[:, 1]
        cm_array["b"] = colors[:, 2]
        cm_array["a"] = colors[:, 3]

        grasps_array = np.zeros(
            (npoints,),
            dtype=[
                ("grasps", np.float32, (4, 4)),
                ("confidence", np.float32),
                ("widths", np.float32),
            ],
        )
        grasps_array["grasps"] = all_grasps.cpu().numpy()
        grasps_array["confidence"] = confs[:, 0]
        grasps_array["widths"] = widths[:, 0]

        return (grasps_array, cm_array)
