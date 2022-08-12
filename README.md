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

<img src="https://user-images.githubusercontent.com/60956651/183575643-7cd2957d-a701-490b-bcb3-b478e3890ede.png" width="500" height="400"/>

</p>

<p align="center">

<img src="https://user-images.githubusercontent.com/60956651/183576318-c12f872c-369a-4cdd-9e1a-50f8b5ae98e7.png" width="500" height="400"/>

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
 ## 7. [Export](#7-export-1)
 ## 8. [Demo](#8demo)

<br>
<hr/>

# 1. Models

현재 지원하는 모델 종류입니다.
 
 <br>

| Model | Params | Resolution(HxW) | Inference time(ms) | Pretrained weights | Pretrained datasets | mAP |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: |
| EfficientNet-Lite-B0 | 3.01m | 300x300 | 0.019ms | [Link](https://github.com/chansoopark98/Tensorflow-Keras-Object-Detection/releases/download/untagged-b94ecf05cb81011df45c/_0807_efficient_lite_v0_lr0.002_b32_e300_single_gpu_bigger_adam_base-128_best_loss_73.0.h5) | PASCAL VOC | 73.0% |
| EfficientNet-Lite-B0 | 3.01m | 300x300 | 0.019ms | [Link](https://github.com/chansoopark98/Tensorflow-Keras-Object-Detection/releases/download/v1.0.0-alpha/_0809_efficient_lite_v0_human_detection_lr0.002_b32_e300_base64_prior_normal_best_loss.h5) | Human detection | - |

<br>

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

PASCAL VOC의 Precision-Recall graph 시각화 기능을 제공합니다.

eval_voc.py를 한 번 이상 실행해야 합니다.

```bash
cd experiemnts
python draw_prec_rec_curve.py
```

<br>

<p align="center">

<img src="https://user-images.githubusercontent.com/60956651/184134684-1cd5f636-8929-4fbc-9f05-d932add8d100.png">

</p>



<br>



**2. COCO2017 EVALUATE**

추가 예정입니다.





<hr>

# 6. Predict
Web-camera 또는 저장된 비디오를 실시간으로 추론할 수 있습니다. <br>
<br>


**Web-cam 실시간 추론**

    python predict_webcam.py


<br>

<br>
<hr>

# 7. Export

학습된 모델을 다양한 프레임워크로 export하는 기능을 제공합니다.

예를들어, tensorflow model을 ONNX로 변환할 수 있으며, 역으로 ONNX 모델을 tensorflow 모델로 재변환합니다.

## **Tensorflow model to another frameworks**

- ### 7.1 Convert to <u>**tensorRT**</u>
- ### 7.2 Convert to <u>**frozen graph**</u>
- ### 7.3 Convert to <u>**ONNX**</u>
- ### 7.4 Convert to <u>**tensorflow_js**</u>
- ### 7.5 Convert to <u>**tensorflow_lite**</u>

## **ONNX model to tensorflow**

- ### 7.6 Convert <u>**ONNX**</u> to <u>**tf saved model + frozen graph**</u>

<br>

<hr>

## **7.1** Convert to tensorRT

<br>

tensorRT를 변환하기 위해서는 tensorRT 엔진을 빌드해야 합니다.

본 레포지토리에서는 tf-trt를 이용하여 tensorRT 엔진을 빌드합니다.

CUDA, CuDNN, TensorRT files


<br>

현재 작성된 코드 기준으로 사용된 CUDA 및 CuDNN 그리고 TensorRT version은 다음과 같습니다. <br>
클릭 시 설치 링크로 이동합니다. <br>
CUDA 및 CuDNN이 사전에 설치가 완료된 경우 생략합니다.

<br>

### CUDA : **[CUDA 11.1](https://www.tensorflow.org/datasets/catalog/overview)**
### CuDNN : **[CuDNN 8.1.1](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.1.1.33/11.2_20210301/cudnn-11.2-linux-x64-v8.1.1.33.tgz)**
### TensorRT : **[TensorRT 7.2.2.3](https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/7.2.2/tars/tensorrt-7.2.2.3.ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz)**

<br>

### **7.1.1** Install TensorRT
<br>

가상 환경을 활성화합니다. (Anaconda와 같이 가상환경을 사용하지 않은 경우 생략합니다)
    
    conda activate ${env_name}

<br>

TensorRT를 설치한 디렉토리로 이동하여 압축을 해제하고 pip를 업그레이드 합니다.

```bash
tar -xvzf TensorRT-7.2.2.3.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz
pip3 install --upgrade pip
```

편집기를 이용하여 배시 쉘에 접근하여 환경 변수를 추가합니다.

```bash
sudo gedit ~/.bashrc
export PATH="/usr/local/cuda-11.1/bin:$PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/park/TensorRT-7.2.2.3/onnx_graphsurgeon
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-11.1/lib64:/usr/local/cuda/extras/CUPTI/lib64:/home/park/TensorRT-7.2.2.3/lib"
```

TensorRT 파이썬 패키지를 설치합니다.

```bash
cd python
python3 -m pip install tensorrt-7.2.2.3-cp38-none-linux_x86_64.whl

cd ../uff/
python3 -m pip install uff-0.6.9-py2.py3-none-any.whl

cd ../graphsurgeon
python3 -m pip install graphsurgeon-0.4.5-py2.py3-none-any.whl

cd ../onnx_graphsurgeon
python3 -m pip install onnx_graphsurgeon-0.2.6-py2.py3-none-any.whl
```

terminal을 열어서 설치가 잘 되었는지 확인합니다.

![test_python](https://user-images.githubusercontent.com/60956651/181165197-6a95119e-ea12-492b-9587-a0c5badc73be.png)

<br>

### **7.1.2** Convert to TF-TensorRT

<br>

TF-TRT 변환 작업 전 사전 학습된 **tensorflow saved model (.pb)** 이 필요합니다. <br>
tensorflow saved model이 없는 경우 **7.1.3** 절차를 따르고, 있는 경우에는 **7.1.4**로 넘어가세요.


- ### **7.1.3** tensorflow saved model이 없는 경우
    <br>

    본 레포지토리에서 **train.py**를 통해 학습된 가중치가 있는 경우 graph model로 변환하는 기능을 제공합니다.

    <br>

    **train.py**에서 **--saved_model** argument로 그래프 저장 모드를 활성화합니다. 그리고 학습된 모델의 가중치가 저장된 경로를 추가해줍니다.

        python train.py --saved_model --saved_model_path='your_model_weights.h5'

    변환된 graph model의 기본 저장 경로는 **'./checkpoints/export_path/1'** 입니다.

    ![saved_model_path](https://user-images.githubusercontent.com/60956651/181168185-376880d3-b9b8-4ea7-8c76-be8b498e34b1.png)

    <br>

- ### **7.1.4** Converting

    **(.pb)** 파일이 존재하는 경우 아래의 스크립트를 실행하여 변환 작업을 수행합니다.

        python convert_to_tensorRT.py ...(argparse options)

    TensorRT 엔진을 통해 모델을 변환합니다. 고정된 입력 크기를 바탕으로 엔진을 빌드하니 스크립트 실행 전 **--help** 인자를 확인해주세요.

    <br>
    
    아래와 같은 옵션을 제공합니다. <br>

    **모델 입력 해상도** (--image_size), **.pb 파일 디렉토리 경로** (input_saved_model_dir) <br>
    
    **TensorRT 변환 모델 저장 경로** (output_saved_model_dir), **변환 부동소수점 모드 설정** (floating_mode)

    <br>


<br>

<hr>

## 7.2 Convert to frozen graph

<br>

다양한 환경에 쉽게 배포하기 위해 tensorflow-keras model을 frozen graph로 변환합니다.

train.py를 통해 학습된 모델 가중치가 필요합니다. 

모델 가중치 저장 경로 및 백본 이름 등 필요한 arguments를 확인해주세요.

    python convert_frozen_graph.py --help


<center>

![image](https://user-images.githubusercontent.com/60956651/183798749-a9e9dc0d-ecc3-4bf5-bd10-683680835e33.png)

</center>

모델 출력에 post-processing을 추가할 경우 아래와 같은 인자를 추가해주세요.

```bash
python convert_frozen_graph.py --include_postprocess
```

<br>

변환이 완료되면 아래와 같은 경로(기본 저장 경로)에 .pb 파일과 .pbtxt 파일이 생성됩니다.

<br>

<center>

![image](https://user-images.githubusercontent.com/60956651/183799185-160c1f01-1a8d-4ad5-9243-7a51c5879b52.png)

</center>

<br>

<hr>

## 7.3 Convert to ONNX

<br>

학습된 tensorflow model을 ONNX 모델로 변환합니다.

ONNX로 변환하기 위해서 7.2 step의 frozen graph 변환 과정을 수행해야 합니다.

<br>

```bash
pip install tf2onnx
 
python -m tf2onnx.convert --input ./your_frozen_graph.pb --output ./frozen_to_onnx_model.onnx --inputs x:0 --outputs Identity:0 --opset 13
```

<br>

제공되는 변환 옵션은 다음과 같습니다.
<br>

--input : Frozen graph model 저장 경로
<br> 
--output : ONNX 모델 저장 경로
<br>
--inputs : Frozen graph 모델 입력 namespace
<br>
--outputs : Frozen graph 모델 출력 namespace
<br>
--opset : ONNX version
<br>

아래와 같이 변환하는 경우,

```bash
python -m tf2onnx.convert --input ./checkpoints/converted_frozen_graph/frozen_graph.pb --output ./checkpoints/converted_frozen_graph/onnx_model.onnx --inputs x:0 --outputs Identity:0 --opset 13
```

<br>

ONNX 모델 파일(.onnx)이 생성됩니다.
<center>

![image](https://user-images.githubusercontent.com/60956651/183800918-17c0b839-aee8-4454-a13d-0c10e904c31f.png)

</center>

<br>

<hr>

## 7.4 Convert to tensorflow_js

<br>

Web (javascript)에서 추론이 가능하도록 tensorflow_js 컨버팅 기능을 제공합니다.

**7.2 step의 frozen graph 변환 작업을 먼저 해야합니다.**

<br>

```bash
tensorflowjs_converter your_frozen_graph.pb ./output_dir/ --input_format=tf_frozen_model --output_node_names='Identity'
```

추가 변환 옵션은 --help로 확인할 수 있습니다.

변환 시 양자화를 하는 경우 --quantize_float16 를 추가합니다.

<br>

```bash
tensorflowjs_converter ./checkpoints/converted_frozen_graph/frozen_graph.pb ./checkpoints/converted_tfjs/ --input_format=tf_frozen_model --output_node_names='Identity' --quantize_float16 
```

변환 결과는 다음과 같습니다.

<center>

![image](https://user-images.githubusercontent.com/60956651/183803331-0d9ed0f4-a3be-4fde-ac8c-6a2c9616a6fb.png)

</center>

<br>

tensorflow-js로 모델 용량에 비례하여 바이너리 파일(.bin)과 모델 정보를 포함하는 model.json 파일이 생성됩니다.

실제 웹에서 추론 가능한 샘플 코드는 다음과 같습니다.

HTML 페이지에서 tensorflow_js를 import 합니다.

<br>

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.19.0/dist/tf.min.js"></script>
```

입력 데이터는 video element 또는 canvas의 이미지를 입력으로 사용합니다.

학습된 모델의 이미지 크기에 맞게 조정합니다.

```javascript
const model = await tf.loadGraphModel('model.json');
const inputImageTensor = tf.expandDims(tf.cast(tf.browser.fromPixels(videoElement), 'float32'), 0);
const resizedImage = tf.image.resizeBilinear(inputImageTensor, [300, 300]);
const normalizedImage = tf.div(resizedImage, 255);

// post-processing이 포함된 경우 모델 최종 출력의 shape은 (N, 6) 입니다.
// N은 검출된 샘플의 개수
// 각 샘플마다 다음과 같은 데이터[x_min, y_min, x_max, y_max, scores, labels]를 포함합니다.
var output = await model.executeAsync(normalizedImage);

output = tf.squeeze(output, 0); // [Batch, N, 6] -> [N, 6]
    
var boxes = output.slice([0, 0], [-1, 4]); // [N, 4]
var scores = output.slice([0, 4], [-1, 1]); // [N, 1]
var labels = output.slice([0, 5], [-1, 1]); // [N, 1]

// 메모리 해제
tf.dispose(output);
tf.dispose(boxes);
tf.dispose(scores);
tf.dispose(labels);
```

<br>

<hr>

## 7.5 Convert to tensorflow_lite

<br>

모바일 Android, ios, raspberry pi와 같은 edge device에서 고속 추론이 가능하도록 tflite 변환 기능을 제공합니다.

양자화를 적용하는 경우 변환 옵션은 다음과 같습니다.

<br>

**GPU** : float16

**CPU** : int8 (TODO)

<br>


```bash
python convert_to_tflite.py --checkpoint_dir='./checkpoints/' \
                            --model_weights='your_model_weights.h5' \
                            --backbone_name='efficient_lite_v0' \
                            --num_classes=21 \
                            --export_dir='./checkpoints/tflite_converted/' \
                            --tflite_name='tflite.tflite'
```
<br>

변환이 완료된 경우 저장 경로에 .tflite 파일이 생성됩니다.

<br>
<center>

![image](https://user-images.githubusercontent.com/60956651/183825152-d887e0d4-b5e6-412c-b0b9-2ce3978de756.png)

</center>

변환 확인을 위해 스크립트를 실행합니다.

```bash
python convert_to_tflite.py --export_dir='./checkpoints/tflite_converted/' \
                            --tflite_name='tflite.tflite' \
                            --test
```

<br>

<hr>

## 7.6 Convert ONNX to tf saved model + frozen graph

<br>

외부 프레임워크에서 학습된 모델 (e.g. pytorch)을 tensorflowjs, tflite 등  웹 및 엣지 디바이스에서
쉽게 추론할 수 있도록 변환 기능을 제공합니다.

<br>

**ONNX로 컨버팅된 모델 파일(.onnx)이 필요합니다!**

```bash
python convert_onnx_to_tf.py --onnx_dir='your_onnx_model.onnx' \
                             --output_dir='onnx2tf_converted'
```

<br>

<hr>

<br>

# 8.Demo

Single image inference test, Human detection 등 다양한 task의 detection demo를 제공합니다.

<br>

## 8.1 Single image inference test

학습된 가중치를 이용하여 단일 이미지에 대한 추론 테스트를 지원합니다.

데모 실행 절차는 다음과 같습니다. <br>

<br>

1. README.md 상단에 있는 PASCAL VOC 데이터셋으로 사전 학습된 EfficientNet-Lite-B0 모델 가중치를 다운로드 받습니다.

2. 저장받은 가중치를 레포지토리의 'checkpoints/' 경로를 복사한 뒤, 파이썬 스크립트를 실행합니다. <br>
   여기서 --weight_name을 저장받은 케라스 가중치 파일 (.h5)에 맞게 변경합니다.

```bash
python predict_image.py --backbone_name='efficient_lite_v0' --batch_size=1 --num_classes=21 --image_dir='./inputs/' --image_format='div' --weight_name='download_your_weights_name.h5'
```

### Caution : 자세한 옵션은 python predict_image.py --help를 통해 확인해주세요.

<br>
<br>

## 8.2 Human detection

PASCAL VOC 07+12, COCO2017에서 human(person) class에 해당하는 샘플만 추출하여 학습한 모델입니다.

총 클래스 수 : 2 (background + human)

1. README.md 상단에 있는 Human detection 데이터셋으로 사전 학습된 EfficientNet-Lite-B0 모델 가중치를 다운로드 받습니다.

2. 단일 이미지 추론 모드는 predict_image.py를 이용하여 추론을 수행합니다. <br>

```bash
python predict_image.py --backbone_name='efficient_lite_v0' --batch_size=1 --num_classes=2 --image_dir='./inputs/' --image_format='div' --weight_name='download_your_weights_name.h5'
```

3. 동영상 추론

```bash
python predict_webcam.py --backbone_name='efficient_lite_v0' --num_classes=2 --image_format='div' --weight_name='download_your_weights_name.h5'
```

<p align="center">

<img src='./experiments/human_Detection_output.gif'>

</p>