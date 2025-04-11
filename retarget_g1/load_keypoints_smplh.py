'''
    NOTE: cannot operate under python3.8+,
    pls refer to humanml3d to create a virtual env with python3.7
'''

import os
import torch
import argparse
import numpy as np
import plotly.graph_objs as go
import numpy as np

import time
import pickle
import numpy as np

from scipy.spatial.transform import Rotation as R
from concurrent.futures import ThreadPoolExecutor, as_completed
from human_body_prior.body_model.body_model import BodyModel


DATA_ROOT_FOLDER = '/data/HumanML3D/smplh_edit_data/'
OUTPUT_FOLDER = '/data/UH1/humanoid_keypoint_g1/temp/'
BETAS_PATH = './betas_param/betas_r5_smplh_g1_neu.npy'


def organize_data_list(data_root):
    data_path_list = []
    for root, _, files in os.walk(data_root):
        for file in files:
            if file.endswith('.npy'):
                data_path_list.append(os.path.join(root, file))
    print(f'Total number of files in the folder: {len(data_path_list)}')
    return data_path_list

def process_batch_keypoints(file_paths, betas_param, device='cpu'):
    smplh_path = "./human_model/smplh/neutral/model.npz"
    dmpls_path = "./human_model/dmpls/neutral/model.npz"
    num_betas = 10
    num_dmpls = 8
    body_model = BodyModel(
        bm_fname=smplh_path,
        num_betas=num_betas,
        num_dmpls=num_dmpls,
        dmpl_fname=dmpls_path
    ).to(device)

    total_frames, outputs, file_names = [], [], []
    stacked_global_orient, stacked_body_pose, stacked_transl = [], [], []

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        file_name = file_name[: -4]            # remove the suffix '.npz'
        data = np.load(file_path, allow_pickle=True)
        data = data.item()
        print('Current processing file name:', file_name)

        bdata_poses = data['poses']
        bdata_trans = data['trans']
        # body_parms = {
        #     'root_orient': torch.Tensor(bdata_poses[:, :3]).to(device),
        #     'pose_body': torch.Tensor(bdata_poses[:, 3:66]).to(device),
        #     'pose_hand': torch.Tensor(bdata_poses[:, 66:]).to(device),
        #     'trans': torch.Tensor(bdata_trans).to(device),
        #     'betas': torch.Tensor(betas_param, dtype=torch.float32).unsqueeze(0).repeat(len(bdata_trans), 1).to(device),
        #     # torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=len(bdata_trans), axis=0)).to(comp_device),
        # }

        body_pose_raw = bdata_poses[:, 3:66]
        global_orient_raw = bdata_poses[:, :3]
        root_trans_raw = bdata_trans
        
        num_frames = body_pose_raw.shape[0]

        try:
            assert global_orient_raw.shape == (num_frames, 3)
            assert body_pose_raw.shape == (num_frames, 63)
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
        pose_body=stacked_body_pose,
        root_orient=stacked_global_orient,
        trans=stacked_transl
    )
    print('Body model forward done!')
    
    # select keypoint joint translations
    joints = body_model_output.Jtr.detach().cpu().numpy().reshape(total_frames[-1], -1, 3)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_root', type=str, default='data/retarget')
    # parser.add_argument('--output_root', type=str, default='data/retarget_smpl')
    # parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--folder', type=int, default=3)
    parser.add_argument('--slice', type=int, default=0)
    args = parser.parse_args()
    device = args.device

    ''' Pre-retargeting Process '''
    
    # get data list from data root folder
    motions_list = organize_data_list(data_root=DATA_ROOT_FOLDER)

    # split the list into 5 parts and select the part wanted
    def split_list(lst, k):
        return [lst[i:i + k] for i in range(0, len(lst), k)]
    motions_list = split_list(motions_list, 20000)
    print(f'Number of slices to have sub-processes: {len(motions_list)}')
    _slice = args.slice
    motions_list = motions_list[_slice]

    motions_list = split_list(motions_list, 5)
    print(f'Number of slices in the list: {len(motions_list)}')

    # generate the data used for retargeting
    betas_param = np.load(BETAS_PATH, allow_pickle=True)
    # betas_param = betas_param.item()
    betas_param = torch.from_numpy(betas_param)

    _to_save_name = 'keypoint_labels_' + str(args.folder) + '_' + str(_slice) + '.pkl'
    keypoint_save_path = os.path.join(OUTPUT_FOLDER, _to_save_name)

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