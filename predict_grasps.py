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
from typing import List, Optional, Tuple

import numpy as np
import rospkg
import torch

# Third-party
from kornia.geometry.conversions import (
    QuaternionCoeffOrder,
    rotation_matrix_to_quaternion,
)
from omegaconf import OmegaConf
from pytorch3d.ops import sample_farthest_points
from scipy.spatial.transform import Rotation

from .tsgrasp.net.lit_tsgraspnet import LitTSGraspNet
from .tsgrasp_utils import (
    PyGrasps,
    PyPose,
    bound_point_cloud_world,
    build_6dof_grasps,
    downsample_xyz,
    eul_to_rotm,
    generate_color_lookup,
    infer_grasps,
    model_metadata_from_yaml,
    transform_to_camera_frame,
)

# Initialize the ROS package manager
rospack = rospkg.RosPack()
pkg_root = rospack.get_path("grasp_synthesis")

# Suppresses a userwarning from kornia
warnings.filterwarnings("ignore", category=UserWarning)


class GraspPredictor:
    """
    Node implementing tsgrasp network
    """

    def __init__(self, yaml_file_path: str, verbose: bool) -> None:
        self.device = torch.device("cuda")
        # Load metadata from yaml file:
        model_metadata = model_metadata_from_yaml(
            os.path.join(pkg_root, yaml_file_path)
        )

        if model_metadata is None:
            print("No Metadata Loaded!")

        # Default params, if not in metadata
        self.confidence_threshold = 0.0
        self.gripper_depth = 0.090
        self.offset_distance = 0.10
        self.downsample = 0.50
        self.queue_len = 1
        self.top_k = 500
        self.outlier_threshold = 0.00005
        self.pts_per_frame = 45000
        self.bounds_min = [0, -0.4, 0.15]
        self.bounds_max = [2, -0.05, 2]
        self.world_bounds = torch.Tensor([[0, -0.4, 0.15], [2, -0.05, 2]]).to(
            self.device
        )
        self.verbose = verbose

        try:
            model_path = os.path.join(pkg_root, model_metadata["ckpt_path"])
            assert os.path.isfile(model_path)

            # Other model params from YAML:
            self.confidence_thresh = model_metadata["confidence_threshold"]
            self.gripper_depth = model_metadata["gripper_depth"]
            self.offset_distance = model_metadata["offset_distance"]
            self.downsample = model_metadata["downsample"]
            self.queue_len = model_metadata["queue_len"]
            self.top_k = model_metadata["top_k"]
            self.outlier_threshold = model_metadata["outlier_threshold"]
            self.pts_per_frame = model_metadata["pts_per_frame"]
            model_cfg = OmegaConf.create(model_metadata["model"]["model_cfg"])
            training_cfg = OmegaConf.create(model_metadata["training"])

        except KeyError as ex:
            print(f"Key Error: {ex}")

        self.color_lookup = generate_color_lookup()

        self.pc_input_msg = None
        self.tf_trans = [0, 0, 0]
        self.tf_rot = [0, 0, np.pi / 2]

        self.pl_model = LitTSGraspNet(model_cfg=model_cfg, training_cfg=training_cfg)
        self.pl_model.load_state_dict(torch.load(model_path)["state_dict"])

        # # load Pytorch Lightning network
        self.pl_model.to(self.device)
        self.pl_model.eval()

    def update_bounds(self, min_pt, max_pt):
        """
        Updates the bounding box for detection

        Args:
            min_pt (np.array): [min_x, min_y, min_z]
            max_pt (np.array): [max_x, max_y, max_z]
        """
        if min_pt is None or max_pt is None:
            pass
        else:
            self.world_bounds = torch.Tensor(np.vstack((min_pt[:3], max_pt[:3]))).to(
                self.device
            )

    @torch.inference_mode()
    def detect(
        self,
        pointcloud: np.array,
        cam_transform: np.array,
        bounds_min=None,
        bounds_max=None,
    ):
        """
        Run grasp prediction on a single input

        Args:
            pointcloud (np.array): _description_
            cam_transform (np.array): _description_
            bounds_min (Optional[np.array]): _description_
            bounds_max (Optional[np.array]): _description_

        Returns:
            _type_: _description_
        """
        # First update the bounds
        if bounds_min is not None and bounds_max is not None:
            self.update_bounds(bounds_min, bounds_max)

        q = [(pointcloud, torch.Tensor(cam_transform))]
        try:
            orig_pts, poses = list(zip(*q))
            pts = [
                torch.from_numpy(pt.astype(np.float32)).to(self.device)
                for pt in orig_pts
            ]
            poses = torch.Tensor(np.stack(poses)).to(self.device, non_blocking=True)
        except ValueError as ex:
            print(f"Is this error because there are fewer than 300x300 points? - {ex}")
            return None, None

        pts = downsample_xyz(pts, self.pts_per_frame)
        if pts is None or any(len(pcl) == 0 for pcl in pts):
            if self.verbose:
                print("No points found after downsampling!")
            return None, None

        pts = bound_point_cloud_world(pts, poses, self.world_bounds)
        if pts is None or any(len(pcl) == 0 for pcl in pts):
            if self.verbose:
                print("No points found after bound_point_cloud_world!")
            return None, None

        pts = transform_to_camera_frame(pts, poses)
        if pts is None or any(len(pcl) < 2 for pcl in pts):
            if self.verbose:
                print("No points found after transform_to_camera_frame!")
            return None, None  # bug with length-one pcs

        grasps, confs, widths = self.identify_grasps(pts)
        all_confs = confs.clone()  # keep the pointwise confs for plotting later

        grasps, confs, widths = self.filter_grasps(grasps, confs, widths)

        if grasps is None:
            if self.verbose:
                print("No points found after filter_grasps!")
            return None, None

        try:
            grasps = self.ensure_grasp_y_axis_upward(grasps)
            grasps = self.transform_to_eq_pose(grasps)
            grasps_data = self.generate_grasps(grasps, confs, widths)
            pc_confidence_data = self.generate_pc_data(pts, all_confs)
            return grasps_data, pc_confidence_data

        except RuntimeError as ex:
            print(f"Encountered Runtime Error! {ex}")
            return None, None

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

    def filter_grasps(self, grasps, confs, widths):
        """
        Initial filtering of grasps based on confidence_threshold

        Args:
            grasps (torch.Tensor): (N, 4, 4) grasp pose tensor
            confs (torch.Tensor): (N, 1) confs tensor
            widths (torch.Tensor): (N, 1) widths tensor

        Returns:
            grasps, confs, widths: Filtered grasps, confs, widths
        """
        # confidence thresholding
        idcs = confs.squeeze() > self.confidence_threshold
        grasps = grasps[idcs]
        confs = confs.squeeze()[idcs]
        widths = widths.squeeze()[idcs]

        if grasps.shape[0] == 0 or confs.shape[0] == 0:
            return None, None, None

        # top-k selection
        vals, top_idcs = torch.topk(
            confs.squeeze(),
            k=min(100 * self.top_k, confs.squeeze().numel()),
            sorted=True,
        )
        grasps = grasps[top_idcs]
        confs = confs[top_idcs]
        widths = widths[top_idcs]

        if grasps.shape[0] == 0:
            return None, None, None

        # furthest point sampling by position
        pos = grasps[:, :3, 3]
        _, selected_idcs = sample_farthest_points(pos.unsqueeze(0), K=self.top_k)
        selected_idcs = selected_idcs.squeeze()

        grasps = grasps[selected_idcs]
        confs = confs[selected_idcs]
        widths = widths[selected_idcs]

        return grasps, confs, widths

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

    def get_orbital_pose(self, poses: List[PyPose]) -> List[PyPose]:
        """
        Generates a orbital pose from an offset.


        Args:
            poses (List[Pose]): Input grasps (poses)

        Returns:
            List[Pose]: Offset poses
        """
        orbital_poses = []
        for pose in poses:
            pos = np.array(pose.position)
            rot_matrix = Rotation.from_quat(pose.orientation).as_matrix()
            z_hat = rot_matrix[:, 2]
            pos = pos - z_hat * self.offset_distance
            o_pose = PyPose(position=pos, orientation=pose.orientation)
            orbital_poses.append(o_pose)
        return orbital_poses

    def generate_grasps(self, grasps, confs, widths) -> PyGrasps:
        """
        Get grasps as dataclass

        Args:
            grasps (torch.Tensor):
            confs (torch.Tensor):
            widths (torch.Tensor):

        """
        qs = (
            rotation_matrix_to_quaternion(
                grasps[:, :3, :3].contiguous(), order=QuaternionCoeffOrder.XYZW
            )
            .cpu()
            .numpy()
        )
        vs = grasps[:, :3, 3].cpu().numpy()

        poses = [PyPose(position=v, orientation=q) for q, v in zip(qs, vs)]
        orbital_poses = self.get_orbital_pose(poses)
        grasps_msg = PyGrasps(poses, orbital_poses, confs.tolist(), widths.tolist())

        return grasps_msg

    def generate_pc_data(self, pts, all_confs) -> np.array:
        """
        Returns point cloud of the grasps with confidences colormapped

        Args:
            pts (torch.Tensor): x, y, z points
            all_confs (torch.Tensor): float values for each grasp
        """
        cloud_points = pts[-1]
        downsample = 4

        confs_downsampled = all_confs[::downsample].cpu().numpy()
        int_confs = np.round(confs_downsampled * 255).astype(np.uint8).squeeze()
        points_downsampled = cloud_points[::downsample].cpu().numpy()
        colors = self.color_lookup[int_confs]
        npoints = len(points_downsampled)
        points_arr = np.zeros(
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
        points_arr["x"] = points_downsampled[:, 0]
        points_arr["y"] = points_downsampled[:, 1]
        points_arr["z"] = points_downsampled[:, 2]
        points_arr["r"] = colors[:, 0]
        points_arr["g"] = colors[:, 1]
        points_arr["b"] = colors[:, 2]
        points_arr["a"] = colors[:, 3]
        return points_arr
