import mujoco, cv2
import numpy as np
import os

''' MUJOCO MODEL INITIALIZATION '''
model = mujoco.MjModel.from_xml_path(r"./robots/h1_2/scene.xml")
data = mujoco.MjData(model)

ROOT_POS = np.array([0.0000, 0.0000, 1.2000])
ROOT_ORI = np.array([0.0000, 0.0000, 0.0000, 1.0000])
DOF_POS = np.array([
    0.0000, -0.3200,  0.0000,  0.5000, -0.1800,  0.0000,
    0.0000, -0.3200,  0.0000,  0.5000, -0.1800,  0.0000,
    0.0000,
    0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
    0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000
])

FILE_PATH = "output"
file_list = ["Play_Violin.npy"] # TODO
file_motions_list = []
for file_name in file_list:
    file_motions_list.append(np.load(f"{FILE_PATH}/{file_name}"))

def set_qpos_init(root_pos=ROOT_POS, root_ori=ROOT_ORI, dof_pos=DOF_POS):
    qpos = np.zeros(34)
    qpos[0: 3] = root_pos
    qpos[3: 7] = root_ori[..., [3, 0, 1, 2]]
    qpos[7:  ] = dof_pos
    return qpos

def set_qpos(root_pos=ROOT_POS, root_ori=ROOT_ORI, dof_pos=DOF_POS):
    qpos = np.zeros(34)
    qpos[0: 3] = root_pos
    qpos[3: 7] = root_ori[..., [3, 0, 1, 2]]
    qpos[7:  ] = dof_pos

    return qpos

if __name__ == '__main__':
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

    for file_name, motions_list in zip(file_list, file_motions_list):
        for i in range(len(motions_list)):
            motions = motions_list[i]
            frames = []

            data.qpos = set_qpos_init()
            mujoco.mj_resetData(model, data)

            with mujoco.Renderer(model, 480, 640) as renderer:
                for j, motion in enumerate(motions):
                    dof_pos = motion[:27]
                    root_pos = motion[27:30]
                    root_ori = motion[30:]
                    data.qpos = set_qpos(root_pos=root_pos, root_ori=root_ori, dof_pos=dof_pos)

                    mujoco.mj_step(model, data)
                    
                    renderer.update_scene(data, scene_option=scene_option)
                    pixels = renderer.render()
                    frames.append(pixels)

            frame_height, frame_width, _ = frames[0].shape
            fps = 30  # Frames per second
            output_file = f'{FILE_PATH}/{file_name[:-4]}{i}.mp4'

            # Create a VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec ('mp4v' for MP4)
            out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

            # Write frames to the video
            for frame in frames:
                out.write(frame[...,::-1])

            # Release the VideoWriter object
            out.release()

            print(f"Video saved as {output_file}")
