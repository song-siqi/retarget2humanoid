import time
import torch
import numpy as np
import mujoco
import mujoco.viewer
import os
import argparse

# from motion_lib_retarget import MotionLibRetarget

MOTION_SOURCE = "./retarget_output/Conductor_smpl.npy"

''' MUJOCO MODEL INITIALIZATION '''
model = mujoco.MjModel.from_xml_path(r"./humanoid_model/g1/g1_29dof_rev_1_0.xml")
# model = mujoco.MjModel.from_xml_path(r"./humanoid_model/g1/g1_29dof_kp_rev_1_0.xml")
data = mujoco.MjData(model)

MOTION_ID = 0

# ROOT_POS = np.array([-1.0936,  0.5604,  0.9308])
ROOT_POS = np.array([0.0000, 0.0000, 1.2000])
# ROOT_ORI = np.array([-1.0538e-02, -6.2510e-02,  3.0003e-02,  9.9754e-01])
ROOT_ORI = np.array([0.0000, 0.0000, 0.0000, 1.0000])
DOF_POS = np.array([
    0.0000, -0.3200,  0.0000,  0.5000, -0.1800,  0.0000,
    0.0000, -0.3200,  0.0000,  0.5000, -0.1800,  0.0000,
    0.0000,  0.0000,  0.0000,
    0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
    0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000
])

def set_qpos_init(root_pos=ROOT_POS, root_ori=ROOT_ORI, dof_pos=DOF_POS):
    qpos = np.zeros(36)
    qpos[0: 3] = root_pos
    qpos[3: 7] = root_ori[..., [3, 0, 1, 2]]
    qpos[7:  ] = dof_pos
    # qpos[2] += 0.1
    return qpos

def set_qpos(root_pos=ROOT_POS, root_ori=ROOT_ORI, dof_pos=DOF_POS):
    qpos = np.zeros(36)
    qpos[0: 3] = root_pos
    qpos[3: 7] = root_ori[..., [3, 0, 1, 2]]
    qpos[7:  ] = dof_pos

    return qpos

if __name__ == '__main__':
    motion_data = np.load(MOTION_SOURCE, allow_pickle=True)
    # motion_data.shape
    # >>> (length, 36)

    # import pdb; pdb.set_trace()
    # exit()

    try:
        # mujoco.mj_resetData(model, data)
        # qpos_set = data.qpos.copy()
        # print('qpos.shape:', qpos_set.shape)
        # index = 0

        ''' DYNAMIC SIMULATION '''
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Close the viewer automatically after 300 wall-seconds.
            start = time.time()

            index = 0

            data.qpos = set_qpos_init()
            mujoco.mj_resetData(model, data)

            while viewer.is_running() and time.time() - start < 10000:
                step_start = time.time()

                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.

                dof_pos = motion_data[:, 0: 29][index].copy()
                root_pos = motion_data[:, 29: 32][index].copy()
                root_ori = motion_data[:, 32: 36][index].copy()
                root_pos[2] += 1.

                data.qpos = set_qpos(root_pos=root_pos, root_ori=root_ori, dof_pos=dof_pos)
                index += 1
                index %= motion_data.shape[0]
                # data.qpos[7: ] = set_qpos()[7: ]
                # data.qpos = set_qpos()

                mujoco.mj_step(model, data)

                # Example modification of a viewer option: toggle contact points every two seconds.
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                # time_until_next_step = model.opt.timestep - (time.time() - step_start)
                time_until_next_step = 1.0 / 20.0
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    except KeyError as error:
        print(error)