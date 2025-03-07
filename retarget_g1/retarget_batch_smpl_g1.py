import os
import torch
import argparse
import yaml
import pdb

import time
import pickle
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
from concurrent.futures import ThreadPoolExecutor, as_completed

# DATA_ROOT_FOLDER1 = '/data/kinematic_smpl/'
# DATA_ROOT_FOLDER1 = '/data/siqisong_share/tease_smpl/'
# DATA_ROOT_FOLDER1 = '/data/siqisong_share/demo_smpl_1105/'
# DATA_ROOT_FOLDER1 = '/data/siqisong_share/pipeline_demo_data_1108/'
# DATA_ROOT_FOLDER1 = '/data/UH1/human_pose/youtube/'
DATA_ROOT_FOLDER1 = './retarget_input/'
DATA_ROOT_FOLDER2 = '/data/charades_smpl/'
DATA_ROOT_FOLDER3 = '/data/kinematic_smpl_2/'

BETAS_PATH = './betas_param/betas_param_r5_default_tpose_smpl.npy'
# OUTPUT_FOLDER = '/data/siqisong_share/smpl_scale_loaded/'
# OUTPUT_FOLDER = '/data/siqisong_share/tease_smpl/'
# OUTPUT_FOLDER = '/data/siqisong_share/demo_smpl_1105/'
# OUTPUT_FOLDER = '/data/siqisong_share/pipeline_demo_data_1108/'
# OUTPUT_FOLDER = '/data/UH1/humanoid_keypoint/temp/'
OUTPUT_FOLDER = './retarget_output/'

# RESULT_FOLDER = '/data/siqisong_share/smpl_scale_retarget/'
# RESULT_FOLDER = '/data/siqisong_share/tease_retargeted/'
# RESULT_FOLDER = '/data/siqisong_share/demo_result_1105/'
# RESULT_FOLDER = '/data/siqisong_share/pipeline_demo_retarget_1108/'
# RESULT_FOLDER = '/data/UH1/humanoid_keypoint/youtube/'
RESULT_FOLDER = './retarget_output/'

BETAS_PATH = './betas_param/betas_r5_smpl_g1_neu2.npy' # './betas_param/betas_param_r5_robot_tpose.npy'
HUMANOID_PATH = './humanoid_model/g1/g1_29dof_rev_1_0.xml'
DEVICE='cpu'

def organize_data_list(data_root):
    data_path_list = []
    for root, _, files in os.walk(data_root):
        for file in files:
            if file.endswith('.npz'):
                data_path_list.append(os.path.join(root, file))
    print(f'Total number of files in the folder: {len(data_path_list)}')
    return data_path_list

def process_batch_keypoints(file_paths, betas_param=None, device='cpu'):
    body_model = SMPL(
        model_path=Path("./human_model/smpl/SMPL_NEUTRAL.pkl"),
        batch_size=1,
        device=device
    )

    total_frames, outputs, file_names = [], [], []
    stacked_global_orient, stacked_body_pose, stacked_transl = [], [], []

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        file_name = file_name[: -4]            # remove the suffix '.npz'
        data = np.load(file_path, allow_pickle=True)
        print('Current processing file name:', file_name)
        
        # print(data.files)
        # >>> ['__key_strict__', '__data_len__', '__keypoints_compressed__', 'smpl', 'pred_cams', 'bboxes_xyxy', 'image_path', 'person_id', 'frame_id']
        # print(data['smpl'].item().keys())
        # >>> dict_keys(['body_pose', 'global_orient', 'betas', 'root_transl'])
        
        smpl_data = data['smpl'].item()
        
        body_pose_raw = smpl_data['body_pose']
        global_orient_raw = smpl_data['global_orient']
        root_trans_raw = smpl_data['root_transl']

        num_frames = body_pose_raw.shape[0]

        try:
            assert body_pose_raw.shape == (num_frames, 23, 3)
            assert global_orient_raw.shape == (num_frames, 3)
            assert root_trans_raw.shape == (num_frames, 3)
        except:
            print('Error in the shape of the data! Skipped')
            continue
        
        # align the axis of the rotation matrix
        rotation = R.from_rotvec(global_orient_raw)
        global_orient_matrix_raw = rotation.as_matrix()
        # M = np.array(
        #     [
        #         [0, 0, 1],
        #         [1, 0, 0],
        #         [0, 1, 0]])
        # rot_matrices_zxy_raw = np.einsum('ijk,kl->ijl', global_orient_matrix_raw, M)
        rot_matrices_zxy_raw = global_orient_matrix_raw[:, [2, 0, 1], :][:, :, [2, 0, 1]]

        root_ori_rotvec = R.from_matrix(rot_matrices_zxy_raw).as_rotvec()
        root_ori_quat = R.from_matrix(rot_matrices_zxy_raw).as_quat()
        # print(R.from_rotvec(global_orient_raw[::50]).as_quat())
        # print(root_ori_quat[::50])

        # import pdb; pdb.set_trace()

        # saving the processed data
        output = dict()
        output['frame_rate'] = 20
        output['time_length'] = (num_frames - 1) / 20
        output['root_translation'] = root_trans_raw[:, [2, 0, 1]]
        output['root_orient'] = root_ori_quat
        output['file_name'] = file_name
        # output['keypoint_trans'] = keypoint_traj
        outputs.append(output)
        file_names.append(file_name)

        total_frames.append(num_frames) # real sampled frames

        global_orient = global_orient_raw.reshape(num_frames, -1)
        body_pose_raw = body_pose_raw.reshape(num_frames, -1)
        transl        = root_trans_raw.reshape(num_frames, -1)
        # print(global_orient.shape, body_pose_raw.shape, transl.shape)
        stacked_global_orient.append(global_orient)
        stacked_body_pose.append(body_pose_raw)
        stacked_transl.append(transl)

    num_files_available = len(total_frames)
    print(f'Number of files available: {num_files_available}')

    stacked_global_orient = np.concatenate(stacked_global_orient, axis=0)
    stacked_body_pose = np.concatenate(stacked_body_pose, axis=0)
    stacked_transl = np.concatenate(stacked_transl, axis=0)

    stacked_global_orient = torch.tensor(stacked_global_orient, dtype=torch.float32, device=device)
    stacked_body_pose = torch.tensor(stacked_body_pose, dtype=torch.float32, device=device)
    stacked_transl = torch.tensor(stacked_transl, dtype=torch.float32, device=device)

    total_frames = np.cumsum(total_frames) #cumsum

    body_model = body_model.to(device)

    betas_param = betas_param.reshape(1, 10).expand(total_frames[-1], 10).to(device)
    print('Forward the body model...')
    body_model_output = body_model(
        betas=betas_param,
        body_pose=stacked_body_pose,
        global_orient=stacked_global_orient,
        transl=stacked_transl
    )
    print('Body model forward done!')
    
    # select keypoint joint translations
    joints = body_model_output.joints.detach().cpu().numpy().reshape(total_frames[-1], -1, 3)
    index_keypoints = [
        1, 4, 7,            # left hip, knee, ankle
        2, 5, 8,            # right hip, knee, ankle
        16, 18, 20,         # left shoulder, elbow, wrist
        17, 19, 21          # right shoulder, elbow, wrist
    ]
    joints_keypoints = joints[:, index_keypoints, :]
    for i, output in enumerate(outputs):
        start_index = 0 if i == 0 else total_frames[i - 1]
        end_index = total_frames[i]
        output['keypoint_trans'] = joints_keypoints[start_index: end_index][:, :, [2, 0, 1]]

    return outputs

def process_keypoints_wrapper(motions, betas_param, device):
    outputs = process_batch_keypoints(
        file_paths=motions,
        # body_model=body_model,
        betas_param=betas_param,
        device=device
    )
    return outputs

def process_batch_motion(keypoint_results, device='cpu'):

    outputs = keypoint_results
    print(f"Retargeting {len(outputs)} motions")
    retarget = BatchG1RetargetKeypoint(
        motion_gts=outputs,
        device=device
    )

    joint_pos, root_trans, root_ori, frames = retarget.retarget() # this returns a stacked tensor of many files
    joint_pos = joint_pos.detach().cpu().numpy()
    root_trans = root_trans.detach().cpu().numpy()
    root_ori = root_ori.detach().cpu().numpy()
    for i in range(len(frames)):
        start = 0 if i == 0 else frames[i - 1]
        end = frames[i]
        num_frames = end - start
        root_trans = outputs[i]['root_translation']
        root_pos = root_trans
        root_rot = R.from_rotvec(root_ori[start:end]).as_quat()  # Convert to quaternion
        dof_pos = joint_pos[start:end]

        full_robot_pos = np.concatenate([dof_pos, root_pos, root_rot], axis=1)

        output = {
            'file_name': outputs[i]["file_name"],
            'fps': outputs[i]["frame_rate"],
            'time_length': outputs[i]["time_length"],
            'num_frames': num_frames,
            'root_pos': root_pos,
            'root_rot': root_rot,  # Convert to rotvec if needed in forward_trackable in poselib
            'dof_pos': dof_pos,
        }
        output_filename = outputs[i]["file_name"]
        output_path = os.path.join(RESULT_FOLDER, output_filename + '.npy')
        print(f"Saving retargeted motion to {output_path}")
        # np.save(output_path, output)
        np.save(output_path, full_robot_pos)

class Config: pass

class G1RetargetKeypoint:
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
        self.robot.num_joints = 29
        self.robot.num_bodies = 30

    def retarget(self):
        print('Total frames to retarget with:', self.data.num_frames)
        # initialize the data with the data input
        joint_pos = torch.zeros(self.data.num_frames, 29, device=self.device, dtype=torch.float32, requires_grad=True)
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
        keypoint_trans[:, 0] = output_trans[:, 2]      # left_hip 
        keypoint_trans[:, 1] = output_trans[:, 4]      # left_knee
        keypoint_trans[:, 2] = output_trans[:, 6]      # left_ankle
        keypoint_trans[:, 3] = output_trans[:, 8]      # right_hip
        keypoint_trans[:, 4] = output_trans[:, 10]     # right_knee
        keypoint_trans[:, 5] = output_trans[:, 12]     # right_ankle
        keypoint_trans[:, 6] = output_trans[:, 17]     # left_shoulder
        keypoint_trans[:, 7] = output_trans[:, 19]     # left_elbow
        keypoint_trans[:, 8] = output_trans[:, 22]     # left_hand
        keypoint_trans[:, 9] = output_trans[:, 24]     # right_shoulder
        keypoint_trans[:, 10] = output_trans[:, 26]    # right_elbow
        keypoint_trans[:, 11] = output_trans[:, 29]    # right_hand      

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
            12, 13, 14,             # torso
            15, 16, 17,             # left_shoulder, 15 is left_shoulder_keypoint_joint
            18,                     # left_elbow, 18 is left_elbow_keypoint_joint
            19, 20, 21,             # left_wrist
            22, 23, 24,             # right_shoulder, 24 is right_shoulder_keypoint_joint
            25,                     # right_elbow, 27 is right_elbow_keypoint_joint
            26, 27, 28,             # right_wrist
        ]
        dof_vel = dof_vel[:, dof_index_from_forward]

        output_trans = output['global_translation'][0, :, :, :]
        keypoint_trans = torch.zeros(batch_len, 12, 3, dtype=torch.float32, device=self.device)
        keypoint_trans[:, 0] = output_trans[:, 2]      # left_hip 
        keypoint_trans[:, 1] = output_trans[:, 4]      # left_knee
        keypoint_trans[:, 2] = output_trans[:, 6]      # left_ankle
        keypoint_trans[:, 3] = output_trans[:, 8]      # right_hip
        keypoint_trans[:, 4] = output_trans[:, 10]     # right_knee
        keypoint_trans[:, 5] = output_trans[:, 12]     # right_ankle
        keypoint_trans[:, 6] = output_trans[:, 17]     # left_shoulder
        keypoint_trans[:, 7] = output_trans[:, 19]     # left_elbow
        keypoint_trans[:, 8] = output_trans[:, 22]     # left_hand
        keypoint_trans[:, 9] = output_trans[:, 24]     # right_shoulder
        keypoint_trans[:, 10] = output_trans[:, 26]    # right_elbow
        keypoint_trans[:, 11] = output_trans[:, 29]    # right_hand
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
        output_path = os.path.join('./visualizations', output_name)
        fig.write_html(output_path)
        print(f"3D scatter plot saved as '{output_name}'")
        pass  

class BatchG1RetargetKeypoint(G1RetargetKeypoint):
    def __init__(
            self,
            motion_gts,
            device=DEVICE
        ):
        self.device = device

        # load from data
        self.nsamples = len(motion_gts)
        self.dt = []
        self.frames = []
        self.init_gt(motion_gts)

        # H1m parameters
        self.robot = Config()
        self.init_robot(mjcf_file=HUMANOID_PATH)

        # retargeting parameters
        self.num_iterations = 2001
    
    def init_gt(self, motion_gts):
        '''
            The motion_gt here is not the path for the motion, but the keypoint translation ground truth
        '''
        root_translation, root_orient, keypoint_trans = [], [], []
        for i, motion_gt in enumerate(motion_gts):
            print(f"Loaded motion data from {motion_gt['file_name']}")
            # data_gt = np.load(motion_gt, allow_pickle=True)
            # data_gt = data_gt.item()
            # print(data_gt.keys())
            num_frames = motion_gt['root_translation'].shape[0]
            root_translation.append(motion_gt['root_translation'])
            root_orient.append(motion_gt['root_orient'])
            keypoint_trans.append(motion_gt['keypoint_trans'])

            self.dt.append(1.0 / motion_gt['frame_rate'])
            self.frames.append(num_frames)

        self.frames = np.cumsum(self.frames) #cumsum
        self.root_translation = np.concatenate(root_translation, axis=0)
        self.root_orient = np.concatenate(root_orient, axis=0)
        self.keypoint_trans = np.concatenate(keypoint_trans, axis=0)
        self.keypoint_gt = torch.from_numpy(self.keypoint_trans).to(self.device).to(torch.float32)

    def retarget(self):
        print('Total frames to retarget with:', self.frames[-1])
        # initialize the data with the data input
        joint_pos = torch.zeros(self.frames[-1], 29, device=self.device, dtype=torch.float32, requires_grad=True)
        root_ori = torch.zeros(self.frames[-1], 3, device=self.device, dtype=torch.float32, requires_grad=True)
        root_trans = torch.zeros(self.frames[-1], 3, device=self.device, dtype=torch.float32, requires_grad=True)

        rotations = R.from_quat(self.root_orient).as_rotvec()
        root_ori_init = torch.from_numpy(rotations).to(self.device).to(torch.float32)
        root_trans_init = torch.from_numpy(self.root_translation).to(self.device).to(torch.float32)

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
                print(f"Iteration {i}: Loss = {loss.item()}, Loss per frame = {loss.item() / self.frames[-1]}")
        
        return joint_pos, root_trans, root_ori, self.frames

    def loss(self, joint_pos, root_trans, root_ori, lamb=0.05):
        loss_smooth = self.loss_smoothing(joint_pos)
        loss_retarget = self.loss_retarget(joint_pos, root_trans, root_ori)
        return lamb * loss_smooth + loss_retarget

    def loss_smoothing(self, joint_pos):
        unsmooth = []
        for i in range(self.nsamples):
            start = 0 if i == 0 else self.frames[i - 1]
            end = self.frames[i]
            last = joint_pos[start: end - 2, :]
            this = joint_pos[start + 1: end - 1, :]
            next = joint_pos[start + 2: end, :]
            unsmooth.append(torch.abs(this - (last + next) * 0.5))
        return torch.sum(torch.cat(unsmooth, dim=0))

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
        error = torch.norm(keypoint - self.keypoint_gt, dim=-1, p=1)
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
        
        output_trans = output['global_translation']

        # Indices corresponding to specific keypoints
        keypoint_indices = [
            2,   # left_hip 
            4,   # left_knee
            6,   # left_ankle
            8,   # right_hip
            10,  # right_knee
            12,  # right_ankle
            17,  # left_shoulder
            19,  # left_elbow
            22,  # left_hand
            24,  # right_shoulder
            26,  # right_elbow
            29,  # right_hand
        ]

        # Create keypoint_trans directly using advanced indexing
        keypoint_trans = output_trans[0, :, keypoint_indices, :]   

        return keypoint_trans


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_root', type=str, default='data/retarget')
    # parser.add_argument('--output_root', type=str, default='data/retarget_smpl')
    # parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--folder', type=str, default='1')
    parser.add_argument('--slice', type=int, default=0)
    args = parser.parse_args()
    device = args.device
    
    ''' Pre-retargeting Process '''
    
    # get data list from data root folder
    if args.folder == '1':
        motions_list = organize_data_list(data_root=DATA_ROOT_FOLDER1)
    elif args.folder == '2':
        motions_list = organize_data_list(data_root=DATA_ROOT_FOLDER2)
    elif args.folder == '3':
        motions_list = organize_data_list(data_root=DATA_ROOT_FOLDER3)
    else:
        raise ValueError('Invalid folder number')
    # motions_list = organize_data_list(data_root=DATA_ROOT_FOLDER1)
    # motions_list = organize_data_list(data_root=DATA_ROOT_FOLDER2)
    
    # split the list into 5 parts and select the part wanted
    def split_list(lst, k):
        return [lst[i:i + k] for i in range(0, len(lst), k)]
    motions_list = split_list(motions_list, 10000)
    print(f'Number of slices to have sub-processes: {len(motions_list)}')
    _slice = args.slice
    motions_list = motions_list[_slice]
    # print(f'Number of files selected: {len(motions_list_selected)}')

    motions_list = split_list(motions_list, 5)
    print(f'Number of slices in the list: {len(motions_list)}')

    # generate the data used for retargeting
    betas_param = np.load(BETAS_PATH, allow_pickle=True)
    # betas_param = betas_param.item()
    betas_param = torch.from_numpy(betas_param)
    
    # body_model = SMPL(
    #     model_path=Path("./human_model/smpl/SMPL_NEUTRAL.pkl"),
    #     batch_size=1,
    #     device=device
    # )
    # _ = process_batch_keypoints(
    #     file_paths=motions_list[0],
    #     betas_param=betas_param,
    #     device=device)
    # exit()

    if args.folder == '1':
        _to_save_name = 'keypoint_labels_' + str(_slice) + '.pkl'
        keypoint_save_path = os.path.join(OUTPUT_FOLDER, _to_save_name)
    elif args.folder == '2':
        keypoint_save_path = os.path.join(OUTPUT_FOLDER, 'keypoint_labels_7.pkl')
    elif args.folder == '3':
        keypoint_save_path = os.path.join(OUTPUT_FOLDER, 'keypoint_labels_8.pkl')
    
    if os.path.exists(keypoint_save_path):
        print("Loading GT keypoint labels ...")
        with open(keypoint_save_path, 'rb') as f:
            keypoints_results = pickle.load(f)
    else:
        print("Creating GT keypoint labels ...")
        keypoint_start_time = time.time()
        keypoints_results = []
        # Create a thread pool
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit tasks to the thread pool
            futures = [
                executor.submit(process_keypoints_wrapper, motion, betas_param, device) 
                for motion in motions_list
            ]

            # As each future completes, merge the results into the list
            for future in as_completed(futures):
                result = future.result()  # This is the (output, file_name) tuple
                keypoints_results.extend(result)

        with open(keypoint_save_path, 'wb') as f:
            pickle.dump(keypoints_results, f)

        keypoint_end_time = time.time()
        keypoint_total_time = keypoint_end_time - keypoint_start_time
        keypoint_avg_time = keypoint_total_time / len(motions_list)
        print(f"GT keypoint labels created in {keypoint_total_time} seconds, processing time per motion: {keypoint_avg_time} seconds")

    # exit()
    print("Retargeting motions ...")
    keypoints_results = split_list(keypoints_results, 100)

    for keypoint_result in tqdm(keypoints_results, total=len(keypoints_results)):
        process_batch_motion(keypoint_result, device=device)