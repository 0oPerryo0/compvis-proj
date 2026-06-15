# Vegetable Image Classification

Training environment for a vegetable image classification project using TensorFlow/Keras and MobileNetV2 transfer learning.

## Regular Project Fit

The slides score a regular project on motivation, methodology/background, demo, lessons learned, and Q&A. This project is shaped around those categories:

- Motivation: supermarket self-checkout and agricultural sorting.
- Methodology/background: image classification, CNNs, convolutions, pooling, ReLU-style nonlinear features, softmax class output, loss/optimization, and selected CNN architectures.
- Demo: predict the vegetable class for test images and a single uploaded image.
- Lessons learned: discuss accuracy, confused classes, dataset quality, lighting, rotation, and background variation.

This code uses MobileNetV2 because it is a convolutional neural network architecture. The final classifier still follows the slide method: images pass through convolutional feature extraction, pooling/aggregation, and a softmax output layer for multiclass classification.

## Folder Layout

Put your Kaggle dataset images into this structure:

```text
data/
  train/
    carrot/
    potato/
    tomato/
  validation/
    carrot/
    potato/
    tomato/
  test/
    carrot/
    potato/
    tomato/
```

Each class name should be a folder. The folder name becomes the label.

## Setup

### Windows PowerShell

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### WSL / Ubuntu GPU Support

Use WSL if you want TensorFlow to train with an NVIDIA GPU. Keep the virtual environment inside Linux home, not inside `/mnt/d`, because Python package installs can hit permission problems on the Windows-mounted drive.

Start WSL from PowerShell:

```powershell
wsl
```

Create and activate a Linux virtual environment:

```bash
mkdir -p ~/venvs
python3 -m venv ~/venvs/compvis
source ~/venvs/compvis/bin/activate
```

Install dependencies with CUDA support:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install "tensorflow[and-cuda]" matplotlib numpy pandas scikit-learn pillow opencv-python kagglehub pypdf
```

Go to the project folder from WSL:

```bash
cd /mnt/d/compvis\ proj
```

Check whether WSL can see the GPU:

```bash
nvidia-smi
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If TensorFlow prints `[]`, set the CUDA library paths after activating the venv:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cublas/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cufft/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/curand/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cusolver/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cusparse/lib
```

If your WSL Python version is not 3.12, replace `python3.12` in those paths with your version from:

```bash
python3 --version
```

Train safely on GPU with a small batch first:

```bash
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_FORCE_GPU_ALLOW_GROWTH=true
python train.py --data-dir data --epochs 3 --fine-tune-epochs 0 --batch-size 4
```

Then run the longer training:

```bash
python train.py --data-dir data --epochs 15 --batch-size 4
```

Monitor GPU usage in another WSL terminal:

```bash
watch -n 1 nvidia-smi
```

Stop training safely with `Ctrl+C`.

## Train

```powershell
python train.py --data-dir data --epochs 15 --batch-size 16
```

To skip fine-tuning and only train the final classifier layer:

```powershell
python train.py --data-dir data --epochs 15 --fine-tune-epochs 0 --batch-size 16
```

To continue training from the saved model:

```powershell
python train.py --data-dir data --epochs 5 --batch-size 16 --resume
```

Only use `--resume` if the class folders are unchanged. If you add, remove, or rename classes, train fresh without `--resume` because the model's output layer must match the dataset classes.

The trained model will be saved to:

```text
models/vegetable_classifier.keras
```

Training graphs and the confusion matrix will be saved to:

```text
outputs/
```

Useful output files for the presentation:

```text
outputs/training_history.png
outputs/confusion_matrix.png
outputs/classification_report.txt
outputs/demo_predictions.png
outputs/class_names.json
```

## Predict One Image

```powershell
python predict.py --image path\to\vegetable.jpg
```

## Webcam Demo

Run the webcam demo after training a model:

```powershell
python webcam_demo.py
```

If the wrong camera opens, try:

```powershell
python webcam_demo.py --camera 1
```

For a stricter prediction threshold:

```powershell
python webcam_demo.py --confidence 0.7
```

Use `Q` to close the webcam window.

The webcam demo is easiest to run from Windows PowerShell because WSL webcam access can require extra setup. The model still works if it was trained in WSL, as long as `models/vegetable_classifier.keras` and `outputs/class_names.json` exist in this project folder.

This is real-time image classification, not object detection. It works best when one clear fruit or vegetable is the dominant object in the frame.

## Recommended Scope

Start with 7-10 vegetable classes and aim for around 80% test accuracy. If accuracy is low, improve the dataset first, then tune the model.
