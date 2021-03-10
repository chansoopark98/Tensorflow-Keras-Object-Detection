# CSNet(Efficient Object Detection Network)
=============
CSnet : Efficient object detection network

이 저장소는 **EfficientNet** 기반 객체 검출 네트워크를 구현하였습니다. 본 구현은 **Tensorflow Keras** 라이브러리 기반 경량화 네트워크입니다. 적은 params와 FLOPS로 경쟁력있는 검출 정확도를 달성하였습니다.

<hr/>

## Table of Contents

 1. [Preferences](#Preferences)
 2. [Install requirements](#Install-requirements)
 3. [Preparing datasets](#Preparing-datasets)
 4. [Train](#Train)
 5. [Eval](#Eval)
 6. [Predict](#Predict)

<hr/>

## Preferences

CSNet은 Tensorflow 기반 코드로 작성되었습니다. 코드는 Windows 및 Linux(Ubuntu) 환경에서 모두 동작합니다.
<table border="0">
<tr>
    <tr>
        <td>
        OS
        </td>
        <td>
        Ubuntu 20.10
        </td>
    </tr>
    <tr>
        <td>
        TF version
        </td>
        <td>
        2.4.1
        </td>
    </tr>
    <tr>
        <td>
        Python version
        </td>
        <td>
        3.8.0
        </td>
    </tr>
    <tr>
        <td>
        CUDA
        </td>
        <td>
        11.1
        </td>
    </tr>
    <tr>
        <td>
        CUDNN
        </td>
        <td>
        cuDNN v8.0.5 (November 9th, 2020), for CUDA 11.1
        </td>
    </tr>
    <tr>
        <td>
        GPU
        </td>
        <td>
        NVIDIA RTX3090 24GB
        </td>
    </tr>
</table>

<hr/>

## Install requirements

학습 및 평가를 위해 Anaconda(miniconda) 가상환경에서 패키지를 다운로드 합니다.
    
    conda create -n envs_name python=3.8

    pip install -r requirements.txt

<hr/>

## Preparing datasets

프로그램에 필요한 dataset은 Tensorflow Datasets 라이브러리(TFDS)를 사용합니다. [TFDS](https://www.tensorflow.org/datasets/catalog/overview)  
COCO_2017, PASCAL_VOC(07+12) 두 개의 datasets을 선택할 수 있습니다.  
<br>
[COCO](https://cocodataset.org/#home)  - Requires 35GB or more storage  
[PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)  - Requires 5GB or more storage    
<br>

download_dataset.py를 사용하여 다운로드 하십시요.  

Download PASCAL VOC Dataset  

    python download_dataset.py --dataset_dir='./datasets/' --train_dataset=voc

Download COCO Dataset  

    python download_dataset.py --dataset_dir='./datasets/' --train_dataset=coco


<hr/>

## Train











 