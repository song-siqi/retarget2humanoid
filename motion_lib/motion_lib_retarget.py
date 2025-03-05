import numpy as np
import os
import yaml

from isaacgym.torch_utils import *
from legged_gym.utils import torch_utils
import torch
import dill as pickle
import pdb
import sys
from tqdm import tqdm
from legged_gym import LEGGED_GYM_ROOT_DIR, MOTION_LIB_DIR

class MotionLibRetarget():
    def __init__(
            self,
            motion_pkl_path=None,
            motion_cfg_path=None,
            motion_folder_path=None,
            device='cpu',
            regen_pkl=False
        ):
        self.device = device

        print("#"*20 + " Loading motion library " + "#"*20)
        if not regen_pkl:
            try:
                self.load_motion_from_pkl(motion_pkl_path)
            except:
                print('Failed to load motion from pkl, generate from npy...')
                print('Setting motion device: cpu')
                self.device = 'cpu'
                self.load_motion_from_npy(
                    motion_cfg_path=motion_cfg_path,
                    motion_folder_path=motion_folder_path
                )
                self.save_motion_to_pkl(motion_pkl_path)
                # exited afterwards
        else:
            print('Regenerating motion pkl...')
            print('Setting motion device: cpu')
            self.device = 'cpu'
            self.load_motion_from_npy(motion_folder_path)
            self.save_motion_to_pkl(motion_pkl_path)
            # exited afterwards

        '''
            NOTE: The 27 dofs refer to the handless h1_2 robot dofs:

                [0: 3]      left_hip_yaw, left_hip_pitch, left_hip_roll
                [3]         left_knee
                [4: 6]      left_ankle_pitch, left_ankle_roll
                [6: 9]      right_hip_yaw, right_hip_pitch, right_hip_roll
                [9]         right_knee
                [10: 12]    right_ankle_pitch, right_ankle_roll
                [12]        torso
                [13: 16]    left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw
                [16: 18]    left_elbow_pitch, left_elbow_roll
                [18: 20]    left_wrist_pitch, left_wrist_yaw
                [20: 23]    right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw
                [23: 25]    right_elbow_pitch, right_elbow_roll
                [25: 27]    right_wrist_pitch, right_wrist_yaw
        '''
            # motion_data['root_pos'] = torch.from_numpy(motion_data['root_pos']).to(self.device).to(torch.float32)
            # motion_data['root_rot'] = torch.from_numpy(motion_data['root_rot']).to(self.device).to(torch.float32)
            # motion_data['dof_pos'] = torch.from_numpy(motion_data['dof_pos']).to(self.device).to(torch.float32)
            # motion_data['root_vel'] = torch.from_numpy(motion_data['root_vel']).to(self.device).to(torch.float32)
            # motion_data['root_ang_vel'] = torch.from_numpy(motion_data['root_ang_vel']).to(self.device).to(torch.float32)
            # motion_data['dof_vel'] = torch.from_numpy(motion_data['dof_vel']).to(self.device).to(torch.float32)
            # motion_data['keypoint_trans'] = torch.from_numpy(motion_data['keypoint_trans']).to(self.device).to(torch.float32)

        motions = self.motions
        self.root_pos       = torch.cat([motion['root_pos'] for motion in motions],         # (total_frames, 3)
                                        dim=0).float().to(self.device)
        self.root_rot       = torch.cat([motion['root_rot'] for motion in motions],         # (total_frames, 4)
                                        dim=0).float().to(self.device)
        self.dof_pos        = torch.cat([motion['dof_pos'] for motion in motions],          # (total_frames, 27)
                                        dim=0).float().to(self.device)
        self.root_vel       = torch.cat([motion['root_vel'] for motion in motions],         # (total_frames, 3)
                                        dim=0).float().to(self.device)
        self.root_ang_vel   = torch.cat([motion['root_ang_vel'] for motion in motions],     # (total_frames, 3)
                                        dim=0).float().to(self.device)
        self.dof_vel        = torch.cat([motion['dof_vel'] for motion in motions],          # (total_frames, 27)
                                        dim=0).float().to(self.device)
        self.keypoint_trans = torch.cat([motion['keypoint_trans'] for motion in motions],   # (total_frames, 12, 3)
                                        dim=0).float().to(self.device)

        lengths = self.motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(dim=0)
        # print(self.motion_fps); quit()
        self.motion_ids = torch.arange(self.num_motions(), dtype=torch.long, device=self.device)
        return

    ######## Functions to be called by simulation class ########
    def num_motions(self):
        return len(self.motions)

    def sample_time(self, motion_ids, truncate_time=None):
        n = len(motion_ids)
        phase = torch.rand(motion_ids.shape, device=self.device)
        # NOTE: this function deals with time matters, NOT the index of frame matters
        motion_len = self.motion_lengths[motion_ids]
        if truncate_time is not None:
            assert truncate_time >= 0.0
            motion_len -= truncate_time
        
        motion_time = motion_len * phase
        motion_time = torch.zeros_like(motion_time)
        return motion_time

    def get_motion_length(self, motion_ids):
        # NOTE: this function deals with time matters, NOT the index of frame matters
        return self.motion_lengths[motion_ids]
    
    def get_motion_state(self, motion_ids, motion_times):
        # num_ids is an array of the motion indexes,
        # which often has a length of the number of envs needed to call this function
        n = len(motion_ids)

        motion_len = self.motion_lengths[motion_ids]
        num_frames = self.motion_num_frames[motion_ids]
        dt = self.motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self.calc_frame_blend(
            time=motion_times, len=motion_len, num_frames=num_frames, dt=dt
        )
        # add the global start frame index to the local frame index
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        root_pos0 = self.root_pos[f0l]
        root_pos1 = self.root_pos[f1l]

        root_rot0 = self.root_rot[f0l]
        root_rot1 = self.root_rot[f1l]

        dof_pos0 = self.dof_pos[f0l]
        dof_pos1 = self.dof_pos[f1l]

        # the velocity of the f0l frame is calculated from f0l and f1l,
        # so there is no need to do the interpolation (already a difference!)
        root_vel = self.root_vel[f0l]
        root_ang_vel = self.root_ang_vel[f0l]
        dof_vel = self.dof_vel[f0l]

        keypoint_trans0 = self.keypoint_trans[f0l]
        keypoint_trans1 = self.keypoint_trans[f1l]

        # check data type
        vals = [root_pos0, root_pos1, root_rot0, root_rot1, dof_pos0, dof_pos1,
                root_vel, root_ang_vel, dof_vel, keypoint_trans0, keypoint_trans1]
        for val in vals:
            assert val.dtype != torch.float64

        # use the slurp function to interpolate spherical data root_rot
        blend = blend.unsqueeze(-1)
        root_pos = (1.0 - blend) * root_pos0 + blend * root_pos1
        root_rot = torch_utils.slerp(root_rot0, root_rot1, blend)
        dof_pos = (1.0 - blend) * dof_pos0 + blend * dof_pos1
        # extend the dimentions to broadcast with keypoint translation data
        blend_exp = blend.unsqueeze(-1)
        keypoint_trans = (1.0 - blend_exp) * keypoint_trans0 + blend_exp * keypoint_trans1

        # NOTE: the data needed is the relevant data of the keybodies from the root_pos
        keypoint_trans = keypoint_trans - root_pos[:, None, :]

        return root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, keypoint_trans

    ######## Utils ########
    def calc_frame_blend(self, time, len, num_frames, dt):
        # NOTE: calculate the local frame index in the trajectory of motion data,
        #       and the blend factor between the two frames,
        #       the local frame_idx0 and frame_idx1 need to add the global start frame index
        phase = time / len
        phase = torch.clamp(phase, 0.0, 1.0)

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = (time - frame_idx0 * dt) / dt

        return frame_idx0, frame_idx1, blend

    def get_total_length(self):
        return sum(self.motion_lengths)

    ######## File Operations ########
    def load_motion_from_npy(
            self,
            motion_cfg_path,
            motion_folder_path
        ):
        with open(motion_cfg_path, 'r') as f:
            motions_list = yaml.load(f, Loader=yaml.SafeLoader)["motions"]
        
        motions_available =[]
        motions_keys = []
        for motion_entry in motions_list.keys():
            if motion_entry == "root":
                continue
            target_motion_file = os.path.join(motion_folder_path, motion_entry + ".npy")
            if os.path.exists(target_motion_file):
                motions_available.append(target_motion_file)
                motions_keys.append(motion_entry)
            else:
                print(f"Motion {motion_entry} is not available, skipping...")
                pass
        print(f'Total number of files detected: {len(motions_available)}')
        num_motions = len(motions_available)

        self.motion_fps = []
        self.motion_dt = []
        self.motion_lengths = []
        self.motion_num_frames = []
        self.motions = []
        # NOTE: is difficulty needed? NO!
        # self.motion_difficulty = []

        for index in tqdm(range(num_motions)):
            motion_file = motions_available[index]
            motion_entry = motions_keys[index]
            print("Loading {:d}/{:d} motion files: {:s}".format(index + 1, num_motions, motion_entry))
            motion_data = np.load(motion_file, allow_pickle=True).item()
            
            self.motion_fps.append(motion_data['fps'])
            dt = 1.0 / motion_data['fps']
            self.motion_dt.append(dt)
            # NOTE: sometimes the length is not accurate directly from the dataset,
            #       which often happens that num_frames = length_from_data * fps + 0.5,
            #       so the length is calculated from the num_frames and fps 
            length = 1.0 / motion_data['fps'] * (motion_data['num_frames'] - 1)
            self.motion_lengths.append(length)
            self.motion_num_frames.append(motion_data['num_frames'])
            self.motions.append(motion_data)
            # NOTE: The reason why not converting the other data to torch:
            # 'root_pos', 'root_rot', 'dof_pos', 'root_vel', 'root_ang_vel', 'dof_vel', 'keypoint_trans'
            # each of them is a np.array of data, so the list is a list of np.array,
            # the transferring to torch means turning the list to a list of torch.tensor,
            # so this work should be done after these data is reloaded from the pkl file
        
        self.motion_fps = torch.from_numpy(np.array(self.motion_fps)).to(self.device).to(torch.float32)
        self.motion_dt = torch.from_numpy(np.array(self.motion_dt)).to(self.device).to(torch.float32)
        self.motion_lengths = torch.from_numpy(np.array(self.motion_lengths)).to(self.device).to(torch.float32)
        self.motion_num_frames = torch.from_numpy(np.array(self.motion_num_frames)).to(self.device).to(torch.int64)

        total_len = self.get_total_length()
        print("Loaded {:d} motions with a total length of {:.3f}s.".format(num_motions, total_len))
        return

    def load_motion_from_pkl(self, motion_pkl_path):
        with open(motion_pkl_path, 'rb') as f:
            objects = pickle.load(f)
        print(f"Loading motions from pkl file: {motion_pkl_path}")

        self.motions = []
        for motion_data in tqdm(objects[0]):
            # print(motion_data.keys())
            # ['fps', 'time_length', 'num_frames',
            #  'root_pos', 'root_rot', 'dof_pos',
            #  'root_vel', 'root_ang_vel', 'dof_vel',
            #  'keypoint_trans']
            # NOTE: this time convert those arrays in the data to torch.tensor
            motion_data['root_pos'] = torch.from_numpy(motion_data['root_pos']).to(self.device).to(torch.float32)
            motion_data['root_rot'] = torch.from_numpy(motion_data['root_rot']).to(self.device).to(torch.float32)
            motion_data['dof_pos'] = torch.from_numpy(motion_data['dof_pos']).to(self.device).to(torch.float32)
            motion_data['root_vel'] = torch.from_numpy(motion_data['root_vel']).to(self.device).to(torch.float32)
            motion_data['root_ang_vel'] = torch.from_numpy(motion_data['root_ang_vel']).to(self.device).to(torch.float32)
            motion_data['dof_vel'] = torch.from_numpy(motion_data['dof_vel']).to(self.device).to(torch.float32)
            motion_data['keypoint_trans'] = torch.from_numpy(motion_data['keypoint_trans']).to(self.device).to(torch.float32)
            self.motions.append(motion_data)
        self.motion_fps = objects[1].to(self.device).to(torch.float32)
        self.motion_dt = objects[2].to(self.device).to(torch.float32)
        self.motion_lengths = objects[3].to(self.device).to(torch.float32)
        self.motion_num_frames = objects[4].to(self.device).to(torch.int64)

        num_motions = self.num_motions()
        total_len = self.get_total_length()
        print("Loaded {:d} motions with a total length of {:.3f}s.".format(num_motions, total_len))
        return

    def save_motion_to_pkl(self, motion_pkl_path):
        objects = [
            self.motions,
            self.motion_fps,
            self.motion_dt,
            self.motion_lengths,
            self.motion_num_frames
        ]
        with open(motion_pkl_path, 'wb') as f:
            pickle.dump(objects, f, pickle.HIGHEST_PROTOCOL)
        print(f"Saved motion library to {motion_pkl_path}")
        exit()
