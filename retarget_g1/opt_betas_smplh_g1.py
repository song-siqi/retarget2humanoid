'''
    NOTE: cannot operate under python3.8+,
    pls refer to humanml3d to create a virtual env with python3.7
'''

import torch
import mujoco
import os
import trimesh

import numpy as np
import plotly.graph_objs as go
import open3d as o3d

from pathlib import Path
import smplx

from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel

PI = np.pi

RANGE_BETAS = 5
TITLE_BETAS = 'betas_r5_smplh_g1_neu'

ROOT_POS = np.array([0.0000, 0.0000, 1.0285])
ROOT_ORI = np.array([1.0000, 0.0000, 0.0000, 0.0000])
# DOF_POS = np.array([
#     0.0000, -0.3200,  0.0000,  0.5000, -0.1800,  0.0000,
#     0.0000, -0.3200,  0.0000,  0.5000, -0.1800,  0.0000,
#     0.0000,  0.0000,  0.0000,
#     0.0000,  PI / 2,  0.0000,  PI / 2,  0.0000,  0.0000,  0.0000,
#     0.0000, -PI / 2,  0.0000,  PI / 2,  0.0000,  0.0000,  0.0000
# ])
DOF_POS = np.array([
    0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
    0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
    0.0000,  0.0000,  0.0000,
    0.0000,  PI / 2,  0.0000,  PI / 2,  0.0000,  0.0000,  0.0000,
    0.0000, -PI / 2,  0.0000,  PI / 2,  0.0000,  0.0000,  0.0000
])

def load_robot_gt_mujoco():
    '''
        Load the keypoint transformation of the default T-pose of H1_2 robot.
    '''
    model = mujoco.MjModel.from_xml_path(r"./humanoid_model/g1/g1_29dof_rev_1_0.xml")
    data = mujoco.MjData(model)
    
    data.qpos[0: 3] = ROOT_POS
    data.qpos[3: 7] = ROOT_ORI
    mujoco.mj_step(model, data)
    pre_xpos = np.copy(data.xpos)

    data.qpos[0: 3] = ROOT_POS
    data.qpos[3: 7] = ROOT_ORI
    data.qpos[7:  ] = DOF_POS
    
    mujoco.mj_step(model, data)
    body_xpos = np.copy(data.xpos)

    # NOTE: if needed to visualize the difference between default pose and our reference T-pose,
    #       please uncomment the line below:
    # visualize_double(pre_xpos, body_xpos, 'T_pose_default_H1_2')

    # keypoint_index = [4, 5, 6, 10, 11, 12, 17, 20, 22, 26, 29, 31]
    keypoint_index = [3, 5, 7, 9, 11, 13, 18, 20, 23, 25, 27, 30]
    keypoint_trans = body_xpos[keypoint_index]
    # print(keypoint_trans.shape)
    
    return keypoint_trans

    # data.xpose is a 33-dim transformation np.ndarray, shaped (33, 3)
    # [0]             root
    # [1]             pelvis
    # [2]             left hip yaw
    # [3]             left hip pitch
    # [4]  [keypoint] left hip roll
    # [5]  [keypoint] left knee
    # [6]  [keypoint] left ankle pitch
    # [7]             left ankle roll
    # [8]             right hip yaw
    # [9]             right hip pitch
    # [10] [keypoint] right hip roll
    # [11] [keypoint] right knee
    # [12] [keypoint] right ankle pitch
    # [13]            right ankle roll
    # [14]            torso
    # [15]            left shoulder pitch
    # [16]            left shoulder roll
    # [17] [keypoint] left shoulder keypoint
    # [18]            left shoulder yaw
    # [19]            left elbow pitch
    # [20] [keypoint] left elbow keypoint
    # [21]            left elbow roll
    # [22] [keypoint] left wrist pitch
    # [23]            left wrist yaw
    # [24]            right shoulder pitch
    # [25]            right shoulder roll
    # [26] [keypoint] right shoulder keypoint
    # [27]            right shoulder yaw
    # [28]            right elbow pitch
    # [29] [keypoint] right elbow keypoint
    # [30]            right elbow roll
    # [31] [keypoint] right wrist pitch
    # [32]            right wrist yaw

def forward_smplh(model, betas, visualize=False):
    body_pose = torch.zeros(1, 63)
    output = model(betas=betas, pose_body=body_pose)

    if visualize:
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        faces = model.faces
        
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)

        o3d.visualization.draw_geometries([mesh_o3d])

    joints = output.Jtr
    keypoint_ids = [1, 4, 7, 2, 5, 8, 16, 18, 20, 17, 19, 21]
    keypoint_trans = torch.zeros(12, 3, dtype=torch.float64)
    keypoint_trans = joints[0][keypoint_ids]
    
    return keypoint_trans

def Loss(model, betas, offset, gt):
    betas = torch.clamp(betas, min=-RANGE_BETAS, max=RANGE_BETAS)
    reindex_ids = [2, 0, 1]
    keypoint_trans = forward_smplh(model, betas)[:, reindex_ids] + offset[None, :]
    keypoint_gt = gt

    return torch.sum(torch.norm(keypoint_trans - keypoint_gt, dim=-1, p=2))

def optim_func(model, param, gt):
    betas = torch.zeros(10)
    betas[0:  5] = param[0:  5]
    betas[5: 10] = param[5: 10]
    return Loss(
        model=model,
        betas=betas.reshape(1, -1),
        offset=param[10: ],
        gt=gt
    )

def visualize_double(keypoint1, keypoint2, title='result'):
    keypoint1 = keypoint1.reshape(-1, 3)
    keypoint2 = keypoint2.reshape(-1, 3)

    scatter1 = go.Scatter3d(
        x=keypoint1[:, 0],
        y=keypoint1[:, 1],
        z=keypoint1[:, 2],
        mode='markers+text',
        text=np.arange(keypoint1.shape[0]),
        marker=dict(
            size=5,
            color='red',
            opacity=0.8
        )
    )

    scatter2 = go.Scatter3d(
        x=keypoint2[:, 0],
        y=keypoint2[:, 1],
        z=keypoint2[:, 2],
        mode='markers+text',
        text=np.arange(keypoint2.shape[0]),
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
            zaxis=dict(title='Z Axis', range=[-3, 3]),
            aspectmode='cube'
        ),
        title=title
    )

    fig = go.Figure(data=[scatter1, scatter2], layout=layout)
    output_name = title + '.html'
    output_path = os.path.join('./visualizations/', output_name)
    fig.write_html(output_path)
    print(f"3D scatter plot saved as '{output_name}'")

if __name__ == '__main__':
    # keypoint_trans_gt = load_robot_gt_mujoco()
    # keypoint_trans_gt = torch.from_numpy(keypoint_trans_gt)
    # print(keypoint_trans_gt)
    # exit()

    keypoint_trans_gt = torch.tensor(
        [
            [ 0.00000000e+00,  1.16452000e-01,  8.95335000e-01],
            [-2.33090489e-06,  1.18600900e-01,  5.89204248e-01],
            [-2.33090489e-06,  1.18506455e-01,  2.71636248e-01],
            [ 0.00000000e+00, -1.16452000e-01,  8.95335000e-01],
            [-2.33090489e-06, -1.18600900e-01,  5.89204248e-01],
            [-2.33090489e-06, -1.18506455e-01,  2.71636248e-01],
            [ 3.73939305e-07,  1.40560443e-01,  1.31746128e+00],
            [ 1.58189156e-02,  3.24275050e-01,  1.32371117e+00],
            [ 5.85427262e-03,  5.08276852e-01,  1.32561040e+00],
            [ 3.73939305e-07, -1.40550443e-01,  1.31746128e+00],
            [ 1.58189156e-02, -3.24265050e-01,  1.32371117e+00],
            [ 5.85427262e-03, -5.08266852e-01,  1.32561040e+00]
        ],
        dtype=torch.float64
    )
    
    num_betas = 10
    num_dmpls = 8

    smplh_path = "./human_model/smplh/neutral/model.npz"
    dmpls_path = "./human_model/dmpls/neutral/model.npz"

    # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')

    neutral_bm = BodyModel(
        bm_fname=smplh_path,
        num_betas=num_betas,
        num_dmpls=num_dmpls,
        dmpl_fname=dmpls_path
    ).to(device)
    
    # body = neutral_bm(
    #     root_orient=torch.zeros(1, 3).to(device),
    #     pose_body=torch.zeros(1, 63).to(device),
    #     trans=torch.zeros(1, 3).to(device),
    #     betas=torch.zeros(1, num_betas).to(device),
    #     # dmpls=torch.zeros(1, num_dmpls).to(device)
    # )
    # print(body.__dict__.keys())
    # print(body.Jtr.shape)
    # test = body.Jtr[: 24]
    # visualize_double(
    #     keypoint1=test.detach().cpu().numpy(),
    #     keypoint2=np.array([0, 0, 0]),
    #     title='smplh_data_visualize'
    # )
    
    x = torch.zeros(13, requires_grad=True)
    # Define the optimizer
    optimizer = torch.optim.Adam([x], lr=0.01)

    # Optimization step
    num_iterations = 6000

    for i in range(num_iterations):
        optimizer.zero_grad()  # Clear the gradients
        loss = optim_func(model=neutral_bm, param=x, gt=keypoint_trans_gt)
        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters
        
        if i % 1000 == 0:
            print(f"Iteration {i}: f(x) = {loss.item()}")
    
    print(f"Result:\nloss: {loss.item()}, betas: {x[0: 10]}, offset:{x[10: ]}")

    betas_title = TITLE_BETAS
    betas = x[0: 10].detach().cpu().numpy()
    betas_path = os.path.join('./betas_param/', betas_title + '.npy')
    np.save(betas_path, betas)

    # NOTE: if needed to visualize the difference between smplx T-pose and robot T-pose,
    #       please uncomment the lines below:
    # betas = x[0: 10].reshape(1, -1)
    # offset = x[10: ]
    # reindex_ids = [2, 0, 1]
    # keypoint_trans = forward_smplh(neutral_bm, betas, visualize=False)[:, reindex_ids] + offset[None, :]
    # visualize_double(
    #     keypoint1=keypoint_trans_gt.detach().cpu().numpy(),
    #     keypoint2=keypoint_trans.detach().cpu().numpy(),
    #     title='h1m_optimized_betas_smplh'
    # )

    # NOTE: if needed to visualize the default pose of the SMPL-X model with the optimized betas,
    #       please uncomment the line below:
    # _ = forward_smplx(model, betas, visualize=True)

'''
    NOTE: SMPL-H model joint definition:
        Pelvis
        Left hip
        Right hip
        Spine1
        Left knee
        Right knee
        Spine2
        Left ankle
        Right ankle
        Spine3
        Left foot
        Right foot
        Neck
        Left collar
        Right collar
        Head
        Left shoulder
        Right shoulder
        Left elbow
        Right elbow
        Left wrist
        Right wrist
'''