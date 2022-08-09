<h1>Tensorflow-Keras Object Detection</h1>

> All about Tensorflow/Keras Object Detection


## Tensorflow/Keras를 활용한 Object detection repository  [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fchansoopark98%2FTensorflow-Keras-Object-Detection&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23C41010&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

<br>




<p align="center">
 <img src="https://img.shields.io/github/issues/chansoopark98/Tensorflow-Keras-Object-Detection">
 <img src="https://img.shields.io/github/forks/chansoopark98/Tensorflow-Keras-Object-Detection">
 <img src="https://img.shields.io/github/stars/chansoopark98/Tensorflow-Keras-Object-Detection">
 <img src="https://img.shields.io/github/license/chansoopark98/Tensorflow-Keras-Object-Detection">
 </p>

<br>

<p align="center">
 <img alt="Python" src ="https://img.shields.io/badge/Python-3776AB.svg?&style=for-the-badge&logo=Python&logoColor=white"/>
 <img src ="https://img.shields.io/badge/Tensorflow-FF6F00.svg?&style=for-the-badge&logo=Tensorflow&logoColor=white"/>
 <img src ="https://img.shields.io/badge/Keras-D00000.svg?&style=for-the-badge&logo=Keras&logoColor=white"/>
 <img src ="https://img.shields.io/badge/OpenCV-5C3EE8.svg?&style=for-the-badge&logo=OpenCV&logoColor=white"/>
 <img src ="https://img.shields.io/badge/Numpy-013243.svg?&style=for-the-badge&logo=Numpy&logoColor=white"/>
</p>

<br>

<p align="center">

<!-- ![12_]() -->

<center><img src="https://user-images.githubusercontent.com/60956651/183575643-7cd2957d-a701-490b-bcb3-b478e3890ede.png" width="500" height="400"/></center>
<center><img src="https://user-images.githubusercontent.com/60956651/183576318-c12f872c-369a-4cdd-9e1a-50f8b5ae98e7.png" width="500" height="400"/></center>

</p>



<br>


### 지원하는 기능
- 데이터 전처리
- Train
- Evaluate
- Predict real-time
- TensorRT 변환
- Tensorflow docker serving

<br>

### **Use library:** 
- Tensorflow
- Tensorflow-js
- Tensorflow-lite
- Tensorflow-datasets
- Tensorflow-addons
- Tensorflow-serving
- Keras
- OpenCV python

### **Options:** Distribute training, Custom Data
### **Models:** Single shot multibox detector (SSD), MobileNet-Series(1,2,3,3+), EfficientNet-Series(V1,V2,Lite)


<br>
<hr/>

# Table of Contents

 ## 1. [Models](#1-models-1)
 ## 2. [Dependencies](#2-dependencies-1)
 ## 3. [Preparing datasets](#3-preparing-datasets-1)
 ## 4. [Train](#4-train-1)
 ## 5. [Eval](#5-eval-1)
 ## 6. [Predict](#6-predict-1)
 ## 7. [Convert TF-TRT](#7-convert-tf-trt-1)
 ## 8. [Tensorflow serving](#8-tensorflow-serving-1)

<br>
<hr/>

# 1. Models

현재 지원하는 모델 종류입니다.
 
 <br>

<table border="0">
<tr>
    <tr>
        <td>
        <h3><strong>Model name</strong></h3>
        </td>
        <td>
        <h3><strong>Params</strong></h3>
        </td>
        <td>
        <h3><strong>Resolution(HxW)</strong></h3>
        </td>
        <td>
        <h3><strong>Inference time(ms)</strong></h3>
        </td>
        <td>
        <h3><strong>Pretrained weights</strong></h3>
        </td>
        <td>
        <h3><strong>Pretrained datasets</strong></h3>
        </td>
    </tr>
    <tr>
        <td>
        EfficientNetLite-SSD
        </td>
        <td>
        3.16m
        </td>
        <td>
        300x300
        </td>
        <td>
        0.019ms
        </td>
        <td>
        TODO
        </td>
        <td>
        voc
        </td>
    </tr> 
</tr>
</table>

## Loss

<table border="0">
<tr>
    <tr>
        <td>
            <h3><strong>Loss</strong></h3>
        </td>
        <td>
            <h3><strong>Implementation</strong></h3>
        </td>
    </tr>
    <!-- CROSS ENTROPY -->
    <tr>
        <td>
            Cross entropy loss
        </td>
        <td>
            OK
        </td>
    </tr>
    <!-- FOCAL CROSS ENTROPY LOSS -->
    <tr>
        <td>
            Focal cross entropy loss
        </td>
        <td>
            OK
        </td>
    </tr>
    <tr>
        <td>
            Hard negative mining
        </td>
        <td>
            OK
        </td>
    </tr>
    <tr>
        <td>
            Smooth L1
        </td>
        <td>
            OK
        </td>
    </tr>

</tr>
</table>

    
<br>
<hr/>

# 2. Dependencies

본 레포지토리의 종속성은 다음과 같습니다.

<table border="0">
<tr>
    <tr>
        <td>
        OS
        </td>
        <td>
        Ubuntu 18.04
        </td>
    </tr>
    <tr>
        <td>
        TF version
        </td>
        <td>
        2.9.1
        </td>
    </tr>
    <tr>
        <td>
        Python version
        </td>
        <td>
        3.8.13
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
        cuDNN v8.1.0 , for CUDA 11.1
        </td>
</table>
<br>
<hr/>

학습 및 평가를 위해 **Anaconda(miniconda)** 가상환경에서 패키지를 다운로드 합니다.
    
    conda create -n envs_name python=3.8

    pip install -r requirements.txt

<br>
<hr/>

# 3. Preparing datasets

프로그램에 필요한 **Dataset**은 **Tensorflow Datasets** 라이브러리([TFDS](https://www.tensorflow.org/datasets/catalog/overview))를 사용합니다. 

### TFDS Object detection dataset
1. [PASCAL VOC](https://www.tensorflow.org/datasets/catalog/voc)
2. [COCO2017](https://www.tensorflow.org/datasets/catalog/coco)

파일 다운로드 방법은 다음과 같습니다.

    # PASCAL VOC download
    python download_datasets.py --train_dataset='voc'

    # COCO2017 download
    python download_datasets.py --train_dataset='coco'

Custom TFDS의 경우 [TFDS 변환 방법](https://github.com/chansoopark98/Tensorflow-Keras-Semantic-Segmentation#3-preparing-datasets-1)을 참고해주세요.



<br>

<hr/>

# 4. Train

학습하기전 tf.data의 메모리 할당 문제로 인해 TCMalloc을 사용하여 메모리 누수를 방지합니다.

    1. sudo apt-get install libtcmalloc-minimal4
    2. dpkg -L libtcmalloc-minimal4

    2번을 통해 설치된 TCMalloc의 경로를 저장합니다


## Training semantic segmentation

**How to RUN?**
    
Single gpu

    LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4.3.0" python train.py

Mutli gpu

    LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4.3.0" python train.py --multi_gpu


### **Caution!**
본 레포지토리는 single-GPU, multi-GPU 환경에서의 학습 및 추론을 지원합니다. <br>
Single-GPU 사용 시, GPU 번호를 설정하여 사용할 수 있습니다. <br>
python train.py --help를 살펴보시고 학습에 필요한 설정값을 argument 인자값으로 추가해주세요. <br>

<br>
<hr>

# 5. Eval
Training 이후 모델의 정확도 평가 및 추론 속도를 계산합니다. <br>
<br>
계산 항목 : FLOPs, MIoU metric, Average inference time
<br>

자세한 항목은 arguments를 참고해주세요.

**1. PASCAL VOC EVALUATE**


    python eval_voc.py --checkpoint_dir='./checkpoints/' --weight_path='weight.h5' --backbone_name='efficient_lite_v0' ... etc

<br>

**2. COCO2017 EVALUATE**

추가 예정입니다.





<hr>

# 6. Predict
Web-camera 또는 저장된 비디오를 실시간으로 추론할 수 있습니다. <br>
<br>


**Web-cam 실시간 추론**

    python predict_realtime.py


<br>

<br>
<hr>

# 7. Convert TF-TRT
고속 추론이 가능하도록 TF-TRT 변환 기능을 제공합니다.
변환에 앞서 tensorRT를 설치합니다.


## 7.1 Install CUDA, CuDNN, TensorRT files

<br>

현재 작성된 코드 기준으로 사용된 CUDA 및 CuDNN 그리고 TensorRT version은 다음과 같습니다. <br>
클릭 시 설치 링크로 이동합니다. <br>
CUDA 및 CuDNN이 사전에 설치가 완료된 경우 생략합니다.

<br>

### CUDA : **[CUDA 11.1](https://www.tensorflow.org/datasets/catalog/overview)**
### CuDNN : **[CuDNN 8.1.1](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.1.1.33/11.2_20210301/cudnn-11.2-linux-x64-v8.1.1.33.tgz)**
### TensorRT : **[TensorRT 7.2.2.3](https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/7.2.2/tars/tensorrt-7.2.2.3.ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz)**

<br>

## 7.2 Install TensorRT
<br>

가상 환경을 활성화합니다. (Anaconda와 같이 가상환경을 사용하지 않은 경우 생략합니다)
    
    conda activate ${env_name}

<br>

TensorRT를 설치한 디렉토리로 이동하여 압축을 해제하고 pip를 업그레이드 합니다.

    tar -xvzf TensorRT-7.2.2.3.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz
    pip3 install --upgrade pip

편집기를 이용하여 배시 쉘에 접근하여 환경 변수를 추가합니다.

    sudo gedit ~/.bashrc
    export PATH="/usr/local/cuda-11.1/bin:$PATH"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/park/TensorRT-7.2.2.3/onnx_graphsurgeon
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-11.1/lib64:/usr/local/cuda/extras/CUPTI/lib64:/home/park/TensorRT-7.2.2.3/lib"

TensorRT 파이썬 패키지를 설치합니다.

    cd python
    python3 -m pip install tensorrt-7.2.2.3-cp38-none-linux_x86_64.whl

    cd ../uff/
    python3 -m pip install uff-0.6.9-py2.py3-none-any.whl

    cd ../graphsurgeon
    python3 -m pip install graphsurgeon-0.4.5-py2.py3-none-any.whl

    cd ../onnx_graphsurgeon
    python3 -m pip install onnx_graphsurgeon-0.2.6-py2.py3-none-any.whl

terminal을 열어서 설치가 잘 되었는지 확인합니다.

![test_python](https://user-images.githubusercontent.com/60956651/181165197-6a95119e-ea12-492b-9587-a0c5badc73be.png)

<br>

## 7.3 Convert to TF-TensorRT

TF-TRT 변환 작업 전 사전 학습된 **graph model (.pb)** 이 필요합니다. <br>
Graph model이 없는 경우 **7.3.1** 절차를 따르고, 있는 경우에는 **7.3.2**로 넘어가세요.


- ### 7.3.1 Graph model이 없는 경우

    본 레포지토리에서 **train.py**를 통해 학습된 가중치가 있는 경우 graph model로 변환하는 기능을 제공합니다.

    **train.py**에서 **--saved_model** argument로 그래프 저장 모드를 활성화합니다. 그리고 학습된 모델의 가중치가 저장된 경로를 추가해줍니다.

        python train.py --saved_model --saved_model_path='your_model_weights.h5'

    변환된 graph model의 기본 저장 경로는 **'./checkpoints/export_path/1'** 입니다.

    ![saved_model_path](https://user-images.githubusercontent.com/60956651/181168185-376880d3-b9b8-4ea7-8c76-be8b498e34b1.png)

    <br>

- ### 7.3.2 Converting

    **(.pb)** 파일이 존재하는 경우 아래의 스크립트를 실행하여 변환 작업을 수행합니다.

        python convert_to_tensorRT.py ...(argparse options)

    TensorRT 엔진을 통해 모델을 변환합니다. 고정된 입력 크기를 바탕으로 엔진을 빌드하니 스크립트 실행 전 **--help** 인자를 확인해주세요.

    <br>
    
    아래와 같은 옵션을 제공합니다. <br>

    **모델 입력 해상도** (--image_size), **.pb 파일 디렉토리 경로** (input_saved_model_dir) <br>
    
    **TensorRT 변환 모델 저장 경로** (output_saved_model_dir), **변환 부동소수점 모드 설정** (floating_mode)

    <br>

<hr>

# 8. Convert to Frozen graph

다양한 환경(ONNX, Tensorflow-js...)에 배포하기 위해 post-processing을 결합하여 Frozen graph로 변환합니다.

    python convert_frozen_graph.py --model_name='efficient_lite_v0' --model_weights='your_model_weights_path'

학습된 모델의 class 개수 설정 등 추가옵션은 arguments를 확인해주세요.