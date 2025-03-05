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

OUTPUT_FOLDER = './retarget_output/'
BETAS_PATH = './betas_param/betas_param_r5_smplx.npy'
HUMANOID_PATH = './humanoid_model/h1_2/h1_2_handless_mark_keypoints.xml'

def process_file(file_path, target_fps=1, smplx_model=None, betas_param=None, visualize=False):
    file_name = os.path.basename(file_path)
    file_name = file_name[: -4]            # remove the suffix '.npz'
    data = np.load(file_path, allow_pickle=True)
    print('Current processing file name:', file_name)

    # downsampling
    curr_fps = data['mocap_frame_rate']
    assert curr_fps % target_fps == 0, 'The target fps should be a divisor of the current fps'
    downsample_ratio = int(curr_fps // target_fps)

    # downsample the root translation
    trans_downsample = data['trans'][:: downsample_ratio]   # downsampling

    # root orientation need to be altered to fit the coordinate system
    root_orient = data['root_orient']
    root_orient = root_orient[:: downsample_ratio]          # downsampling
    rotation = R.from_rotvec(root_orient)
    root_ori_matrix = rotation.as_matrix()
    # transform the orientation matrix [y, z, x] to the correct form [x, y, z]
    root_ori_matrix = root_ori_matrix[:, :, [2, 0, 1]]
    rotation = R.from_matrix(root_ori_matrix)
    root_ori_quat = rotation.as_quat()      # convert to quaternion [x, y, z, w]

    # generate keypoint joint translations from the SMPLX model
    keypoint_traj = np.zeros((data['poses'].shape[0], 12, 3))
    keypoint_traj = keypoint_traj[:: downsample_ratio]      # downsampling
    for index in range(data['poses'].shape[0]):
        if index % downsample_ratio != 0:
            continue

        global_orient = torch.from_numpy(data['root_orient'][index]).reshape(1, -1).to(torch.float32)
        body_pose_raw = torch.from_numpy(data['pose_body'][index]).reshape(1, -1).to(torch.float32)
        transl        = torch.from_numpy(data['trans'][index]).reshape(1, -1).to(torch.float32)

        smplx_output = smplx_model(
            betas=betas_param.reshape(1, 10),
            body_pose=body_pose_raw,
            global_orient=global_orient,
            transl=transl
        )

        # select keypoint joint translations
        joints = smplx_output.joints.detach().cpu().numpy().reshape(-1, 3)
        index_keypoints = [
            1, 4, 7,            # left hip, knee, ankle
            2, 5, 8,            # right hip, knee, ankle
            16, 18, 20,         # left shoulder, elbow, wrist
            17, 19, 21          # right shoulder, elbow, wrist
        ]
        joints_keypoints = joints[index_keypoints, :]

        downsampled_index = index // downsample_ratio
        keypoint_traj[downsampled_index] = joints_keypoints

    # saving the processed data
    output = dict()
    output['frame_rate'] = target_fps
    output['time_length'] = data['mocap_time_length']
    output['root_translation'] = trans_downsample
    output['root_orient'] = root_ori_quat
    output['keypoint_trans'] = keypoint_traj

    if visualize:
        visualize_3d(
            xyz=keypoint_traj[0].reshape(-1, 3),
            file_name=file_name,
            key_name='keypoint',
            write_html=True
        )
    
    return output, file_name

def visualize_3d(xyz, file_name, key_name, write_html=False):
    import plotly.graph_objs as go

    scatter = go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=xyz[:, 2],
            colorscale='Viridis',
            opacity=0.8
        )
    )

    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X Axis', range=[-3, 3]),
            yaxis=dict(title='Y Axis', range=[-3, 3]),
            zaxis=dict(title='Z Axis', range=[-3, 3]),
            aspectmode='cube'
        ),
        title=file_name + '_' + key_name
    )

    fig = go.Figure(data=[scatter], layout=layout)

    if write_html:
        output_name = file_name + '_' + key_name + '.html'
        output_path = os.path.join('./visualizations/', output_name)
        fig.write_html(output_path)
        print(f"3D scatter plot saved as '{output_name}'")

class Config: pass

class H1mRetargetKeypoint:
    def __init__(
            self,
            motion_gt,
            motion_name,
            device='cuda:0'
        ):
        self.device = device

        # load from data
        self.file_name = motion_name
        self.data = Config()
        self.init_gt(motion_gt)

        # H1m parameters
        self.robot = Config()
        self.init_robot(mjcf_file=HUMANOID_PATH)

        # retargeting parameters
        self.num_iterations = 2001
    
    def init_gt(self, motion_gt):
        '''
            The motion_gt here is not the path for the motion, but the 
        '''
        data_gt = motion_gt
        # data_gt = np.load(motion_gt, allow_pickle=True)
        # data_gt = data_gt.item()
        # print(data_gt.keys())
        self.data.frame_rate = data_gt['frame_rate']
        self.data.time_length = data_gt['time_length']
        self.data.num_frames = data_gt['root_translation'].shape[0]
        self.data.root_translation = data_gt['root_translation']
        self.data.root_orient = data_gt['root_orient']
        self.data.keypoint_trans = data_gt['keypoint_trans']

        self.dt = 1.0 / self.data.frame_rate

        print(f"Loaded motion data from {self.file_name}")

        # visualize_3d(
        #     xyz=self.data.keypoint_trans[::100].reshape(-1, 3),
        #     file_name='pre_retarget_test',
        #     key_name='keypoint',
        #     write_html=True
        # )

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
        self.robot.num_joints = 27
        self.robot.num_bodies = 32

    def retarget(self):
        print('Total frames to retarget with:', self.data.num_frames)
        # initialize the data with the data input
        joint_pos = torch.zeros(self.data.num_frames, 27, device=self.device, dtype=torch.float32, requires_grad=True)
        root_ori = torch.zeros(self.data.num_frames, 3, device=self.device, dtype=torch.float32, requires_grad=True)
        root_trans = torch.zeros(self.data.num_frames, 3, device=self.device, dtype=torch.float32, requires_grad=True)

        rotations = R.from_quat(self.data.root_orient).as_rotvec()
        root_ori_init = torch.from_numpy(rotations).to(self.device).to(torch.float32)
        root_trans_init = torch.from_numpy(self.data.root_translation).to(self.device).to(torch.float32)

        root_ori.data = root_ori_init
        root_trans.data = root_trans_init

        optimizer = torch.optim.Adam([joint_pos, root_trans], lr=0.005)

        for i in range(self.num_iterations):
            optimizer.zero_grad()   # Clear the gradients
            loss = self.loss(
                joint_pos=joint_pos,
                root_trans=root_trans,
                root_ori=root_ori
            )
            loss.backward()         # Compute gradients
            optimizer.step()        # Update parameters
            
            if i % 500 == 0:
                print(f"Iteration {i}: Loss = {loss.item()}, Loss per frame = {loss.item() / self.data.num_frames}")
        
        return joint_pos, root_trans, root_ori 

    def loss(self, joint_pos, root_trans, root_ori, lamb=0.05):
        loss_smooth = self.loss_smoothing(joint_pos)
        loss_retarget = self.loss_retarget(joint_pos, root_trans, root_ori)
        return lamb * loss_smooth + loss_retarget

    def loss_smoothing(self, joint_pos):
        last = joint_pos[0: -2, :]
        this = joint_pos[1: -1, :]
        next = joint_pos[2:   , :]
        unsmooth = torch.abs(this - (last + next) * 0.5)
        return torch.sum(unsmooth)

    def loss_retarget(self, joint_pos, root_trans, root_ori):
        '''
            keypoint: (batch_length, num_keypoints, 3)
            keypoint_gt: (batch_length, num_keypoints, 3)
        '''
        keypoint = self.forward(
            joint_pos=joint_pos,
            root_trans=root_trans,
            root_ori=root_ori
        )
        keypoint_gt = torch.from_numpy(self.data.keypoint_trans).to(self.device).to(torch.float32)

        error = torch.norm(keypoint - keypoint_gt, dim=-1, p=1)
        return torch.sum(error)

    def forward(self, joint_pos=None, root_trans=None, root_ori=None):
        '''
            Forward the kinematics model to get the keypoint translation
            Input: 
                joint_pos       (batch_length, num_joints)
            Output:
                keypoint_trans  (batch_length, num_keypoints, 3)
        '''
        batch_len = joint_pos.shape[0]
        # clamp the joint position to fit the joint range
        joint_pos = self.set_clamp(joint_pos)
        # print(joint_pos)

        # reshape the joint_pos to fit the input of the kinematics model
        # the output is the description of the rotations starting from the root body in axis-angle format
        pose_batch = self.joint_pos_to_pose_batch(joint_pos=joint_pos, root_ori=root_ori)

        output = self.robot.kinematics.fk_batch(
            pose=pose_batch,
            trans=root_trans.unsqueeze(0),
            convert_to_mat=True,
            return_full=False)
        
        output_trans = output['global_translation'][0, :, :, :]

        keypoint_trans = torch.zeros(batch_len, 12, 3, dtype=torch.float32, device=self.device)
        keypoint_trans[:, 0] = output_trans[:, 3]      # left_hip 
        keypoint_trans[:, 1] = output_trans[:, 4]      # left_knee
        keypoint_trans[:, 2] = output_trans[:, 5]      # left_ankle
        keypoint_trans[:, 3] = output_trans[:, 9]      # right_hip
        keypoint_trans[:, 4] = output_trans[:, 10]     # right_knee
        keypoint_trans[:, 5] = output_trans[:, 11]     # right_ankle
        keypoint_trans[:, 6] = output_trans[:, 16]     # left_shoulder
        keypoint_trans[:, 7] = output_trans[:, 19]     # left_elbow
        keypoint_trans[:, 8] = output_trans[:, 21]     # left_hand
        keypoint_trans[:, 9] = output_trans[:, 25]     # right_shoulder
        keypoint_trans[:, 10] = output_trans[:, 28]    # right_elbow
        keypoint_trans[:, 11] = output_trans[:, 30]    # right_hand        

        return keypoint_trans
    
    def forward_trackable(self, joint_pos=None, root_trans=None, root_ori=None):
        '''
            Forward the kinematics model to get the keypoint translation to be tracked in Imitation Learning
            Input: 
                joint_pos       (batch_length, num_joints)
            Output:
                keypoint_trans  (batch_length, num_keypoints, 3)
        '''
        batch_len = joint_pos.shape[0]
        # clamp the joint position to fit the joint range
        joint_pos = self.set_clamp(joint_pos)
        # print(joint_pos)

        # reshape the joint_pos to fit the input of the kinematics model
        # the output is the description of the rotations starting from the root body in axis-angle format
        pose_batch = self.joint_pos_to_pose_batch(joint_pos=joint_pos, root_ori=root_ori)

        output = Humanoid_Batch_H1(
            mjcf_file=self.robot.mjcf_file,
            device='cpu',
            extend_hand=False,
            extend_head=False
        ).fk_batch(
            pose=pose_batch.cpu(),
            trans=root_trans.unsqueeze(0).cpu(),
            convert_to_mat=True,
            return_full=True,
            dt=self.dt)
        
        root_pos = root_trans.cpu().numpy()
        root_rot = R.from_rotvec(root_ori.cpu().numpy()).as_quat()  # transfer to quaternion
        dof_pos = joint_pos.cpu().numpy()
        root_vel = output['global_root_velocity'][0].cpu().numpy()
        root_ang_vel = output['global_root_angular_velocity'][0].cpu().numpy()
        dof_vel = output['dof_vels'][0].cpu().numpy()

        dof_index_from_forward = [
            0, 1, 2, 3, 4, 5,       # left_leg
            6, 7, 8, 9, 10, 11,     # right_leg
            12,                     # torso
            13, 14, 16,             # left_shoulder, 15 is left_shoulder_keypoint_joint
            17, 19,                 # left_elbow, 18 is left_elbow_keypoint_joint
            20, 21,                 # left_wrist
            22, 23, 25,             # right_shoulder, 24 is right_shoulder_keypoint_joint
            26, 28,                 # right_elbow, 27 is right_elbow_keypoint_joint
            29, 30                  # right_wrist
        ]
        dof_vel = dof_vel[:, dof_index_from_forward]

        output_trans = output['global_translation'][0, :, :, :]
        keypoint_trans = torch.zeros(batch_len, 12, 3, dtype=torch.float32, device=self.device)
        keypoint_trans[:, 0] = output_trans[:, 3]      # left_hip 
        keypoint_trans[:, 1] = output_trans[:, 4]      # left_knee
        keypoint_trans[:, 2] = output_trans[:, 5]      # left_ankle
        keypoint_trans[:, 3] = output_trans[:, 9]      # right_hip
        keypoint_trans[:, 4] = output_trans[:, 10]     # right_knee
        keypoint_trans[:, 5] = output_trans[:, 11]     # right_ankle
        keypoint_trans[:, 6] = output_trans[:, 15]     # left_shoulder
        keypoint_trans[:, 7] = output_trans[:, 18]     # left_elbow
        keypoint_trans[:, 8] = output_trans[:, 21]     # left_hand
        keypoint_trans[:, 9] = output_trans[:, 24]     # right_shoulder
        keypoint_trans[:, 10] = output_trans[:, 27]    # right_elbow
        keypoint_trans[:, 11] = output_trans[:, 30]    # right_hand
        keypoint_trans = keypoint_trans.detach().cpu().numpy()

        # global_angular_velocity = output['global_angular_velocity'][0]
        # global_velocity = output['global_velocity'][0]
        # local_rotation = output['local_rotation'][0]
        # global_rotation = output['global_rotation'][0]

        return root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, keypoint_trans

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
        pose_batch[0, :, 1, 2] = joint_pos[:, 0]
        pose_batch[0, :, 2, 1] = joint_pos[:, 1]
        pose_batch[0, :, 3, 0] = joint_pos[:, 2]
        pose_batch[0, :, 4, 1] = joint_pos[:, 3]
        pose_batch[0, :, 5, 1] = joint_pos[:, 4]
        pose_batch[0, :, 6, 0] = joint_pos[:, 5]
        # right leg
        pose_batch[0, :, 7, 2] = joint_pos[:, 6]
        pose_batch[0, :, 8, 1] = joint_pos[:, 7]
        pose_batch[0, :, 9, 0] = joint_pos[:, 8]
        pose_batch[0, :, 10, 1] = joint_pos[:, 9]
        pose_batch[0, :, 11, 1] = joint_pos[:, 10]
        pose_batch[0, :, 12, 0] = joint_pos[:, 11]
        # torso
        pose_batch[0, :, 13, 2] = joint_pos[:, 12]
        # left arm
        pose_batch[0, :, 14, 1] = joint_pos[:, 13]
        pose_batch[0, :, 15, 0] = joint_pos[:, 14]
        # pose_batch[0, :, 16] is for left_shoulder_keypoint with fixed joint
        pose_batch[0, :, 17, 2] = joint_pos[:, 15]
        pose_batch[0, :, 18, 1] = joint_pos[:, 16]
        # pose_batch[0, :, 19] is for left_elbow_keypoint with fixed joint
        pose_batch[0, :, 20, 0] = joint_pos[:, 17]
        pose_batch[0, :, 21, 1] = joint_pos[:, 18]
        pose_batch[0, :, 22, 2] = joint_pos[:, 19]
        # right arm
        pose_batch[0, :, 23, 1] = joint_pos[:, 20]
        pose_batch[0, :, 24, 0] = joint_pos[:, 21]
        # pose_batch[0, :, 25] is for right_shoulder_keypoint with fixed joint
        pose_batch[0, :, 26, 2] = joint_pos[:, 22]
        pose_batch[0, :, 27, 1] = joint_pos[:, 23]
        # pose_batch[0, :, 28] is for right_elbow_keypoint with fixed joint
        pose_batch[0, :, 29, 0] = joint_pos[:, 24]
        pose_batch[0, :, 30, 1] = joint_pos[:, 25]
        pose_batch[0, :, 31, 2] = joint_pos[:, 26]

        return pose_batch
    
    def visualize_retarget_result(self, joint_pos, root_trans, root_ori):
        keypoint = self.forward(
            joint_pos=joint_pos,
            root_trans=root_trans,
            root_ori=root_ori
        )
        keypoint_gt = torch.from_numpy(self.data.keypoint_trans).to(self.device).to(torch.float32)
        
        length = keypoint.shape[0]
        interval = int(length / 5)

        keypoint = keypoint.detach().cpu().numpy()[::interval].reshape(-1, 3)
        keypoint_gt = keypoint_gt.detach().cpu().numpy()[::interval].reshape(-1, 3)
        
        scatter = go.Scatter3d(
            x=keypoint[:, 0],
            y=keypoint[:, 1],
            z=keypoint[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color='red',
                opacity=0.8
            )
        )

        scatter_gt = go.Scatter3d(
            x=keypoint_gt[:, 0],
            y=keypoint_gt[:, 1],
            z=keypoint_gt[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color='green',
                opacity=0.8
            )
        )

        layout = go.Layout(
            scene=dict(
                xaxis=dict(title='X Axis', range=[-3, 3]),
                yaxis=dict(title='Y Axis', range=[-3, 3]),
                zaxis=dict(title='Y Axis', range=[-3, 3]),
                aspectmode='cube'
            ),
            title=self.file_name + '_retarget_result'
        )

        fig = go.Figure(data=[scatter, scatter_gt], layout=layout)

        output_name = self.file_name + '_retarget_result' + '.html'
        output_path = os.path.join('./visualize/', output_name)
        fig.write_html(output_path)
        print(f"3D scatter plot saved as '{output_name}'")
        pass  


if __name__ == '__main__':
    # NOTE: this is the input data path
    # NOTE: here we select one example from AMASS/CMU dataset (in SMPLX form) as input
    motion = "./retarget_input/example.npz"

    smplx_model = SMPLX(
        model_path=Path("./human_model/smplx/SMPLX_NEUTRAL.npz"),
        batch_size=1,
        device='cpu'
    )
    
    betas_param = np.load(BETAS_PATH, allow_pickle=True)
    # betas_param = betas_param.item()
    betas_param = torch.from_numpy(betas_param)
    
    output, file_name = process_file(
        file_path=motion,
        target_fps=60,
        smplx_model=smplx_model,
        betas_param=betas_param,
        visualize=False
    )

    retarget = H1mRetargetKeypoint(
        motion_gt=output,
        motion_name=file_name,
        device='cuda:0'
    )
            
    joint_pos, root_trans, root_ori = retarget.retarget()
    root_trans = torch.from_numpy(output['root_translation'])

    root_pos = root_trans.cpu().detach().numpy()
    root_rot = R.from_rotvec(root_ori.cpu().detach().numpy()).as_quat()  # transfer to quaternion
    dof_pos = joint_pos.cpu().detach().numpy()

    output = {
        'file_name': file_name,
        'fps': retarget.data.frame_rate,
        'time_length': retarget.data.time_length,
        'num_frames': retarget.data.num_frames,
        'root_pos': root_pos,
        'root_rot': root_rot,
        'dof_pos': dof_pos,
    }

    output_filename = retarget.file_name
    output_path = os.path.join(OUTPUT_FOLDER, output_filename + '.npy')
    np.save(output_path, output)

    '''
    # NOTE: another version of output which includes more information
    (
        root_pos,
        root_rot,
        dof_pos,
        root_vel,
        root_ang_vel,
        dof_vel,
        keypoint_trans
    ) = retarget.forward_trackable(
        joint_pos=joint_pos.detach(),
        root_trans=root_trans.detach(),
        root_ori=root_ori.detach()
    )

    output = {
        'file_name': file_name,
        'fps': retarget.data.frame_rate,
        'time_length': retarget.data.time_length,
        'num_frames': retarget.data.num_frames,
        'root_pos': root_pos,
        'root_rot': root_rot,
        'dof_pos': dof_pos,
        'root_vel': root_vel,
        'root_ang_vel': root_ang_vel,
        'dof_vel': dof_vel,
        'keypoint_trans': keypoint_trans
    }

    output_filename = retarget.file_name
    output_path = os.path.join(OUTPUT_FOLDER, output_filename + '.npy')
    np.save(output_path, output)
    '''

    '''
    # NOTE: if needs visualization
    retarget.visualize_retarget_result(
        joint_pos=joint_pos,
        root_trans=root_trans,
        root_ori=root_ori
    )
    '''