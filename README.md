# LEHP-DETR
LEHP-DETR: A model with backbone improved and hybrid encoding innovated for flax capsule detection
# LEHP-DETR improvement project based on [ultralytics](https://github.com/ultralytics/ultralytics) and [RT-DETR](https://github.com/lyuwenyu/RT-DETR)

    The version of ultralytics used in this project is 8.0.201, identified by __version__ in ultralytics/__init__.py.

    In order to try to align with the training strategy in the official PaddlePaddle-(RT-DETR), the following changes were made to the source code.
    1. change the parameter max_norm to 0.1 in torch.nn.utils.clip_grad_norm_ in the optimiser_step function in ultralytics/engine/trainer.py. 2. change the parameter max_norm to 0.1 in the optimiser_step function in ultralytics/engine/trainer.py.
    2. in ultralytics/engine/trainer.py, in the _setup_train function, set self.args.nbs equal to self.batch_size, so that the model doesn't need to build up the gradient before updating the parameters.
    3. ultralytics/cfg/default.yaml

# LEHP-DETR environment configuration

    1. Execute pip uninstall ultralytics to uninstall the ultralytics library installed in the environment. <Note here, if you are also using yolov8, it is better to use anaconda to create a virtual environment for this code to use, to avoid environment conflicts that may cause some strange problems>.
    2. uninstallation is complete after the same execution again, if there is WARNING: Skipping ultralytics as it is not installed. prove that it has been uninstalled.
    3. If you need to use the official CLI runtime, you need to install the ultralytics library, execute the command: <python setup.py develop>, of course, after the installation of the code will still be valid.
    4. Additional package installation commands.
        pip install timm==0.9.8 thop efficientnet_pytorch==0.7.1 einops grad-cam==1.4.8 dill==0.3.6 albumentations==1.3.1 pytorch_wavelets==1.3.0 tidecv
        pip install -U openmim
        mim install mmengine
        mim install ‘mmcv>=2.0.0’
