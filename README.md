# retarget2humanoid

The aim of this repo is to retarget from Human Mesh Descriptions (SMPL, SMPL-X, etc) to Humanoid Poses, a partly implementation of the retargeting process in [UH-1 Project](https://usc-gvl.github.io/UH-1/).

## Dependencies

To establish the environment while dealing with digital human models such as SMPL and SMPL-X, we need to create a new environment
```bash
conda create -n retarget python=3.8
pip install -r requirements.txt
```
There might be some other dependencies missing so `requirements.txt` will be further refined.

Due to copyright reasons, the files below needed to be downloaded and added manually:
```
retarget_g1/  # also in retarget_h1_2/
    PoseLib/ ----> refers to https://github.com/T-K-233/PoseLib
    human_model/
        smpl/ ---> refers to [SMPL](https://smpl.is.tue.mpg.de/)
        smplx/ --> refers to [SMPLX](https://smpl-x.is.tue.mpg.de/)
    *other files and folders in the retarget section*
*other files and folders in the repo*
```

**For SMPL**, please download version 1.1.0 for Python 2.7, and rename the `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` as `SMPL_NEUTRAL.pkl`.
**For SMPL-X**, please download the SMPL-X 1.1 version, and keep the `.npz` files.
The file system structure is shown in the file system below
```
retarget/
    PoseLib/
    human_model/
        smpl/
            SMPL_NEUTRAL.pkl
        smplx/ 
            SMPLX_FEMALE.npz
            SMPLX_MALE.npz
            SMPLX_NEUTRAL.npz
    *other files and folders in the retarget section*
*other files and folders in the repo*
```

## Usage

To be constructed...