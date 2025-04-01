import os
import torch
import argparse
import yaml
import pdb

import numpy as np
import torch.nn as nn
import plotly.graph_objs as go

from pathlib import Path
from scipy.spatial.transform import Rotation as R
from smplx import SMPL, SMPLX
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from utils.torch_h1_humanoid_batch import Humanoid_Batch_H1


HUMANOID_PATH = './humanoid_model/g1/g1_29dof_rev_1_0.xml'


class Config: pass

class G1ForwardKeypoint:
    def __init__(
            self,
            device='cpu'
        ):
        self.device = device
        self.dt = 0.05

        # H1m parameters
        self.robot = Config()
        self.init_robot(mjcf_file=HUMANOID_PATH)
    
    def init_robot(self, mjcf_file):
        self.robot.kinematics = Humanoid_Batch_H1(
            mjcf_file=mjcf_file,
            device=self.device,
            extend_hand=False,
            extend_head=False
        )
        # print(self.robot.kinematics.mjcf_data)
        self.robot.mjcf_file = mjcf_file
        self.robot.node_names = self.robot.kinematics.mjcf_data['node_names']
        self.robot.parent_indices = self.robot.kinematics.mjcf_data['parent_indices']
        self.robot.local_translation = self.robot.kinematics.mjcf_data['local_translation'].to(torch.float32).to(self.device)
        self.robot.local_rotation = self.robot.kinematics.mjcf_data['local_rotation'].to(torch.float32).to(self.device)
        self.robot.joints_range = self.robot.kinematics.mjcf_data['joints_range'].to(torch.float32).to(self.device)
        self.robot.num_joints = 29
        self.robot.num_bodies = 30

    def set_clamp(self, joint_pos):
        '''
            Clamp the joint position to the joint range
            joint_pos: (batch_length, num_joints)
        '''
        assert joint_pos.shape[-1] == self.robot.num_joints, "Input joint_pos shape does not match the robot model"
        min_vals = self.robot.joints_range[:, 0]
        max_vals = self.robot.joints_range[:, 1]
        joint_pos = torch.clamp(joint_pos, min=min_vals, max=max_vals)
        return joint_pos

    def joint_pos_to_pose_batch(self, joint_pos, root_ori):
        '''
            Input: joint_pos (batch_length, num_joints)
            Output: pose_batch (1, batch_length, num_bodies, 3)
        '''
        batch_len = joint_pos.shape[0]
        pose_batch = torch.zeros(1, batch_len, self.robot.num_bodies, 3, dtype=torch.float32, device=self.device)
        
        # to convert the joint_pos into rotation vectors of the corresponding bodies in the kinematics model
        # NOTE: added shoulder and elbow keypoint bodies, so there are 28 + 4 = 32 bodies in total
        pose_batch[0, :, 0, :] = root_ori
        # left leg
        pose_batch[0, :, 1, 1] = joint_pos[:, 0]
        pose_batch[0, :, 2, 0] = joint_pos[:, 1]
        pose_batch[0, :, 3, 2] = joint_pos[:, 2]
        pose_batch[0, :, 4, 1] = joint_pos[:, 3]
        pose_batch[0, :, 5, 1] = joint_pos[:, 4]
        pose_batch[0, :, 6, 0] = joint_pos[:, 5]
        # right leg
        pose_batch[0, :, 7, 1] = joint_pos[:, 6]
        pose_batch[0, :, 8, 0] = joint_pos[:, 7]
        pose_batch[0, :, 9, 2] = joint_pos[:, 8]
        pose_batch[0, :, 10, 1] = joint_pos[:, 9]
        pose_batch[0, :, 11, 1] = joint_pos[:, 10]
        pose_batch[0, :, 12, 0] = joint_pos[:, 11]
        # torso
        pose_batch[0, :, 13, 2] = joint_pos[:, 12]
        pose_batch[0, :, 14, 0] = joint_pos[:, 13]
        pose_batch[0, :, 15, 1] = joint_pos[:, 14]
        # left arm
        pose_batch[0, :, 16, 1] = joint_pos[:, 15]
        pose_batch[0, :, 17, 0] = joint_pos[:, 16]
        pose_batch[0, :, 18, 2] = joint_pos[:, 17]
        pose_batch[0, :, 19, 1] = joint_pos[:, 18]
        pose_batch[0, :, 20, 0] = joint_pos[:, 19]
        pose_batch[0, :, 21, 1] = joint_pos[:, 20]
        pose_batch[0, :, 22, 2] = joint_pos[:, 21]
        # right arm
        pose_batch[0, :, 23, 1] = joint_pos[:, 22]
        pose_batch[0, :, 24, 0] = joint_pos[:, 23]
        pose_batch[0, :, 25, 2] = joint_pos[:, 24]
        pose_batch[0, :, 26, 1] = joint_pos[:, 25]
        pose_batch[0, :, 27, 0] = joint_pos[:, 26]
        pose_batch[0, :, 28, 1] = joint_pos[:, 27]
        pose_batch[0, :, 29, 2] = joint_pos[:, 28]

        return pose_batch


if __name__ == '__main__':
    ## NOTE: Phase 1: Test the data structure of 'amass_cmu.pkl'
    import joblib
    with open('test/amass_cmu.pkl', 'rb') as file:
        data = joblib.load(file)

    # import pdb; pdb.set_trace()
    # data[list(data.keys())[0]].keys()
    # >>> dict_keys(['root_trans_offset', 'pose_aa', 'dof', 'root_rot', 'fps'])
    ## NOTE：578 here is the number of frames in the retargeted data
    # data[list(data.keys())[0]]['root_trans_offset'].shape
    # >>> (578, 3)
    # data[list(data.keys())[0]]['pose_aa'].shape
    # >>> (578, 32, 3)
    # data[list(data.keys())[0]]['dof'].shape
    # >>> (578, 29)
    # data[list(data.keys())[0]]['root_rot'].shape
    # >>> (578, 4)


    ## NOTE: Phase 2: Test the data structure of our retargeted data
    data = np.load('retarget_output/Play_baseball_smpl.npy', allow_pickle=True)

    # import pdb; pdb.set_trace()
    # data.shape
    # >>> (73, 36)

    # NOTE: given full_robot_pos = np.concatenate([dof_pos, root_pos, root_rot], axis=1) in retargeting
    dof_pos = data[:, 0:29]
    root_pos = data[:, 29:32]
    root_rot = data[:, 32:36]

    _root_rot = R.from_quat(root_rot).as_rotvec()

    # transfer to torch tensor
    joint_pos = torch.from_numpy(dof_pos).float()
    root_trans = torch.from_numpy(root_pos).float()
    root_ori = torch.from_numpy(_root_rot).float()

    _forward = G1ForwardKeypoint(device='cpu')

    batch_len = joint_pos.shape[0]
    # clamp the joint position to fit the joint range
    joint_pos = _forward.set_clamp(joint_pos)
    # print(joint_pos)

    # reshape the joint_pos to fit the input of the kinematics model
    # the output is the description of the rotations starting from the root body in axis-angle format
    pose_batch = _forward.joint_pos_to_pose_batch(joint_pos=joint_pos, root_ori=root_ori)
            
    output = Humanoid_Batch_H1(
        mjcf_file=_forward.robot.mjcf_file,
        device='cpu',
        extend_hand=False,
        extend_head=False
    ).fk_batch(
        pose=pose_batch.cpu(),
        trans=root_trans.unsqueeze(0).cpu(),
        # convert_to_mat=True,
        # return_full=True,
        dt=_forward.dt)

    # import pdb; pdb.set_trace()
    # output.keys()
    # >>> dict_keys(['global_translation', 'global_rotation_mat', 'global_rotation', 'local_rotation',
    #                'global_root_velocity', 'global_root_angular_velocity', 'global_angular_velocity',
    #                'global_velocity', 'dof_pos', 'dof_vels', 'fps'])

    ## NOTE: The output below is given due to the original processing of the data in OmniH2O repo.
    root_trans_offset_dump = root_trans.clone()
    root_trans_offset_dump[..., 2] -= output.global_translation[..., 2].min().item() - 0.08
    root_trans_offset_dump = root_trans_offset_dump.squeeze().cpu().detach().numpy()

    # dof_pos (np.array) is the retargeting output / model output, there is no need for altering
    dof_dump = dof_pos.copy()

    # in OmniH2O repo:
    #   pose_aa_dump = pose_aa_h1_new.squeeze().cpu().detach().numpy()
    #   pose_aa_h1_new = torch.cat([gt_root_rot[None, :, None], h1_rotation_axis * dof_pos_new, torch.zeros((1, N, 2, 3)).to(device)], axis = 2)
    #   fk_return = h1_fk.fk_batch(pose_aa_h1_new, root_trans_offset[None, ])
    # so pose_aa_dump is the input of fk_batch
    pose_aa_h1_new = pose_batch.clone()
    pose_aa_dump = pose_aa_h1_new.squeeze().cpu().detach().numpy()

    # root_rot_dump = R.from_rotvec(gt_root_rot.cpu().numpy()).as_quat() in OmniH2O repo,
    # while gt_root_rot is _root_rot = R.from_quat(root_rot).as_rotvec() here,
    # so in fact root_rot_dump is the original quat in root_rot
    root_rot_dump = root_rot.copy()

    data_dump = {
        "root_trans_offset": root_trans_offset_dump,
        "pose_aa": pose_aa_dump,
        "dof": dof_dump,
        "root_rot": root_rot_dump,
        "fps": 20
    }
    
    ## NOTE：219 here is the number of frames in the retargeted data
    # print(root_trans_offset_dump.shape)
    # >>> (219, 3)
    # print(pose_aa_dump.shape)
    # >>> (219, 30, 3)
    # print(dof_dump.shape)
    # >>> (219, 29)
    # print(root_rot_dump.shape)
    # >>> (219, 4)