# NeRF as a Non-Distant Environment Emitter in Physics-based Inverse Rendering

![papers_232s3](https://github.com/user-attachments/assets/596658f5-e7d9-4dc8-abb8-d166c3c6b2c5)

### [Project Page](https://nerfemitterpbir.github.io/) | [Paper](https://arxiv.org/abs/2402.04829) | [Dataset](https://github.com/gerwang/nerf-emitter/releases)

## Installation: Setup the environment

### Create environment

Nerfstudio requires `python >= 3.8`. We recommend using conda to manage dependencies. Make sure to install [Conda](https://docs.conda.io/en/latest/miniconda.html) before proceeding.

```bash
conda create --name nerfemitter -y python=3.8
conda activate nerfemitter
pip install --upgrade pip
```

### Dependencies

Install PyTorch with CUDA (this repo has been tested with CUDA 11.8) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).
`cuda-toolkit` is required for building `tiny-cuda-nn`.

For CUDA 11.8:

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" -y cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

### Installing nerf-emitter

```bash
git clone --recursive https://github.com/gerwang/nerf-emitter.git
cd nerf-emitter
pip install --upgrade pip setuptools
pip install -e .
conda install -y ffmpeg
imageio_download_bin freeimage
```

### Add path to `differentiable-sdf-rendering`

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export PYTHONPATH=$PYTHONPATH:differentiable-sdf-rendering/python' > $CONDA_PREFIX/etc/conda/activate.d/setsdfpath.sh
conda deactivate
conda activate nerfemitter
```

## Running Experiments

Download the dataset from the [release](https://github.com/gerwang/nerf-emitter/releases) page and unzip to the project directory.

Please refer to the `scripts/` directory for running the training, mesh export, novel-view synthesis and relighting.

### Synthetic dataset

```bash
bash scripts/synthetic/ours/run_${object}.sh
```

#### Environment map baseline


```bash
bash scripts/synthetic/baseline/run_${object}.sh
```

### Real dataset

#### Our method

```bash
bash scripts/real/ours/run_${object}.sh
```

#### Environment map baseline


```bash
bash scripts/real/baseline/run_${object}.sh
```

## Project Structure

- `differentiable-sdf-rendering/` contains our modified version of the differentiable SDF rendering code
  - `assets/` contains the scene files for mitsuba3
    - `integrator_sdf.xml` is the configuration file for the mitsuba3 SDF renderer
    - `sdf_scene.xml` is the scene file
  - `python/` contains the emitter and integrator plugins for mitsuba3, written in Python
    - `emitters/` contains the emitter plugins
      - `nerf.py` converts emitter queries to NeRF evaluations
      - `nerf_emitter_op.py` wraps NeRF evaluation in PyTorch as a `dr.CustomOp`
      - `vMF.py` contains the implementation of the emitter importance sampling for NeRF
    - `integrators/` contains the integrator plugins
      - `reparam_split_light.py` is the base class which splits one rendering megakernel into two
      - `sdf_curvature.py` computes the curvature loss
      - `sdf_direct_reparam_onesamplemis.py` is the main integrator
    - `sensors/` contains the sensor plugins
      - `spherical_sensor.py` can render a environment map
    - `opt_configs.py` contains the configuration for the optimization
    - `variables.py` contains the optimization SDF and voxel grids
- `nerfstudio/` contains the NeRFStudio code
  - `configs/`
    - `method_configs.py` contains the configuration for `sdf-nerfacto` and `sdf-gt-envmap`
  - `data/`
    - `datamanagers/`
      - `mitsuba_datamanager.py` loads images and mitsuba sensors for inverse rendering
    - `dataparsers/`
      - `instant_ngp_dataparser.py` parses the synthetic dataset
      - `nerfstudio_dataparser.py` parses the real dataset
  - `field_components/`
    - `rotater.py` handles the rotations of the turntable
  - `model_coponents/`
    - `gmm_cluster_light.py` clusters the light point cloud into a Gaussian mixture model
    - `mi_sensor_generators.py` converts NeRFStudio cameras to Mitsuba sensors
    - `output_light_pc.py` uses sampled rays to obtain a NeRF point cloud
  - `models/`
    - `nerfacto.py` is the modified nerf that supports HDR training
    - `sdf_nerfacto.py` supports batch checkpointing
  - `path_guiding/` contains interfaces for importance sampling
    - `path_guiding.py` is the base class
    - `vmf_guiding.py` implements the importance sampling using vMF mixtures
  - `pipelines/`
    - `mitsuba_sdf.py` is the main pipeline for inverse rendering
  - `scripts/`
    - `render.py` renders novel-view and relighted images
    - `train.py` is the training script

# Acknowledgement

This project is based on [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) and [differentiable-sdf-rendering](https://github.com/rgl-epfl/differentiable-sdf-rendering). Thanks for these great projects.