<h1> Human Falling Detection and Tracking </h1>

Using Tiny-YOLO oneclass to detect each person in the frame and use 
[AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) to get skeleton-pose and then use
[ST-GCN](https://github.com/yysijie/st-gcn) model to predict action from every 30 frames 
of each person tracks.

Which now support 7 actions: Standing, Walking, Sitting, Lying Down, Stand up, Sit down, Fall Down.

<div align="center">
    <img src="sample1.gif" width="416">
</div>

## Prerequisites
Tested on: 
- Python > 3.8.10
- Pytorch > 1.18.0

Original test run on: i7-9750H, RTX 2080 Max-Q, CUDA 10.1

## Pre-Trained Models

- Tiny-YOLO oneclass - [.pth](https://drive.google.com/file/d/1obEbWBSm9bXeg10FriJ7R2cGLRsg-AfP/view?usp=sharing),
[.cfg](https://drive.google.com/file/d/19sPzBZjAjuJQ3emRteHybm2SG25w9Wn5/view?usp=sharing)
- SPPE FastPose (AlphaPose) - [resnet101](https://drive.google.com/file/d/1N2MgE1Esq6CKYA6FyZVKpPwHRyOCrzA0/view?usp=sharing),
[resnet50](https://drive.google.com/file/d/1IPfCDRwCmQDnQy94nT1V-_NVtTEi4VmU/view?usp=sharing)
- ST-GCN action recognition - [tsstg](https://drive.google.com/file/d/1mQQ4JHe58ylKbBqTjuKzpwN2nwKOWJ9u/view?usp=sharing)

### Models Directory
Your Models directory should be structured this way:
```
Models
├── sppe
│   ├── fast_res101_320x256.pth
│   └── fast_res50_256x192.pth
├── TSSTG
│   └── tsstg-model.pth
└── yolo-tiny-onecls
    ├── best-model.pth
    └── yolov3-tiny-onecls.cfg
```

## Basic Use

1. Download all pre-trained models into ./Models folder.
2. Run main.py
```
    python main.py ${video file or camera source}
```

## Reference

- AlphaPose : https://github.com/Amanbhandula/AlphaPose
- ST-GCN : https://github.com/yysijie/st-gcn
