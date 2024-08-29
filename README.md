#LEHP-DETR


# LEHP-DETR improvement project based on [ultralytics](https://github.com/ultralytics/ultralytics) and [RT-DETR](https://github.com/lyuwenyu/RT-DETR)

    The version of ultralytics used in this project is 8.0.201, identified by __version__ in ultralytics/__init__.py.

    In order to try to align with the training strategy in the official PaddlePaddle-(RT-DETR), the following changes were made to the source code.
    1. change the parameter max_norm to 0.1 in torch.nn.utils.clip_grad_norm_ in the optimiser_step function in ultralytics/engine/trainer.py. 2. change the parameter max_norm to 0.1 in the optimiser_step function in ultralytics/engine/trainer.py.
    2. in ultralytics/engine/trainer.py, in the _setup_train function, set self.args.nbs equal to self.batch_size, so that the model doesn't need to build up the gradient before updating the parameters.
    3. ultralytics/cfg/default.yaml configuration file changes <see tutorial video for details>.

# LEHP-DETR environment configuration

    1. Execute pip uninstall ultralytics to uninstall the ultralytics library installed in the environment. <Note here, if you are also using yolov8, it is better to use anaconda to create a virtual environment for this code to use, to avoid environment conflicts that may cause some strange problems>.
    2. uninstallation is complete after the same execution again, if there is WARNING: Skipping ultralytics as it is not installed. prove that it has been uninstalled.
    3. If you need to use the official CLI runtime, you need to install the ultralytics library, execute the command: <python setup.py develop>, of course, after the installation of the code will still be valid. (See https://blog.csdn.net/qq_16568205/article/details/110433714 for an explanation of the role of develop.) Note: If you don't need to use the official CLI, you can skip this step.
    4. Additional package installation commands.
        pip install timm==0.9.8 thop efficientnet_pytorch==0.7.1 einops grad-cam==1.4.8 dill==0.3.6 albumentations==1.3.1 pytorch_wavelets==1.3.0 tidecv
        The following are the packages you need to install to use dyhead, if you don't install them successfully, dyhead won't work properly!
        pip install -U openmim
        mim install mmengine
        mim install ‘mmcv>=2.0.0’


# Description of some of the files that come with it
1. train.py
    Script for training the model
2. main_profile.py
    Script to output the model and the parameters and calculations of each layer of the model (rtdetr-l and rtdetr-x can't output the parameters and calculations of each layer and the time because of the problem of the thop library).
3. val.py
    Script to compute metrics using the trained model.
4. detect.py
    Inference script
5. track.py
    Script for tracking inference
6. heatmap.py
    Script to generate heatmap
7. get_FPS.py
    Script to calculate model storage size, model inference time, FPS
8. get_COCO_metrice.py
    Script to calculate COCO metrics
9. plot_result.py
    Script for plotting curve comparisons
