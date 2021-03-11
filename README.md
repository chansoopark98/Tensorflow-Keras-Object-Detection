<h1>CSNet</h1>

**CSnet : Efficient object detection network**

이 저장소는 **EfficientNet** 기반 객체 검출 네트워크를 구현하였습니다. 본 구현은 **Tensorflow Keras** 라이브러리 기반 경량화 네트워크입니다. 적은 params와 FLOPS로 경쟁력있는 검출 정확도를 달성하였습니다.

<hr/>

## Table of Contents

 1. [Preferences](#Preferences)
 2. [Install requirements](#Install-requirements)
 3. [Preparing datasets](#Preparing-datasets)
 4. [Train](#Train)
 5. [Eval](#Eval)
 6. [Predict](#Predict)
 7. [Reference](#Reference)

<hr/>

## Preferences

CSNet은 Tensorflow 기반 코드로 작성되었습니다. 코드는 **Windows** 및 **Linux(Ubuntu)** 환경에서 모두 동작합니다.
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

학습 및 평가를 위해 **Anaconda(miniconda)** 가상환경에서 패키지를 다운로드 합니다.
    
    conda create -n envs_name python=3.8

    pip install -r requirements.txt

<hr/>

## Preparing datasets

프로그램에 필요한 **Dataset**은 **Tensorflow Datasets** 라이브러리(**TFDS**)를 사용합니다. [TFDS](https://www.tensorflow.org/datasets/catalog/overview)  
**COCO_2017**, **PASCAL_VOC(07+12)** 두 개의 datasets을 선택할 수 있습니다.  
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
  
훈련 과정에 앞서 데이터셋을 사전에 준비해야 합니다. 데이터셋을 다운로드 한 후 train.py로 훈련을 시작합니다.  
<br/>
```python
parser.add_argument("--batch_size",     type=int,   help="배치 사이즈값 설정", default=32)
parser.add_argument("--epoch",          type=int,   help="에폭 설정", default=100)
parser.add_argument("--image_size",     type=int,   help="모델 입력 이미지 크기 설정", default=384)
parser.add_argument("--lr",             type=float, help="Learning rate 설정", default=0.001)
parser.add_argument("--model_name",     type=str,   help="저장될 모델 이름", default='MODEL_NAME')
parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/')
parser.add_argument("--backbone_model", type=str,   help="EfficientNet 모델 설정", default='B0')
parser.add_argument("--train_dataset",  type=str,   help="학습에 사용할 dataset 설정 coco or voc", default='voc')
parser.add_argument("--pretrain_mode",  type=bool,  help="저장되어 있는 가중치 로드", default=False)
```  
아래와 같이 실행할 수 있습니다.  

    python train.py --batch_size=32 --epoch=200 --image_size=384 --lr=0.001 --model_name=test model  
    --dataset_dir='./datasets/' --checkpoint_dir='./checkpoints/' --backbone_model=B0 --train_dataset=voc  

사전 저장된 모델을 이어서 훈련하는 경우 아래 인자를 추가하십시요.  

    --pretrain_mode=True
<br>
<hr>

## Eval
훈련 이후 모델 평가를 위해 eval.py를 실행합니다.
```python
parser.add_argument("--image_size",     type=int,   help="모델 입력 이미지 크기 설정", default=384)
parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/model_name.h5')
parser.add_argument("--backbone_model", type=str,   help="EfficientNet 모델 설정", default='B0')
parser.add_argument("--train_dataset",  type=str,   help="학습에 사용할 dataset 설정 coco or voc", default='voc')
```  
아래와 같이 실행할 수 있습니다.  

    python eval.py --image_size=384 --dataset_dir='./datasets/' --checkpoint_dir='./checkpoints/model_name.h5'  
    --backbone_model=B0 --train_dataset=voc  


## Predict
사전 저장된 모델로 이미지 추론을 predict.py로 실행합니다. 테스트에 사용할 이미지 파일과 출력 결과를 저장할 디렉토리를 지정해야 합니다.  

```plain
└── CSNet root
       ├── inputs/  # This is the image directory to use for testing.
       |   ├── image_1.jpg 
       |   └── image_2.jpg
       └── outputs/  # This is the directory to save the inference result image.    
           ├── image_1_output.jpg 
           └── image_2_output.jpg
```  
<br/>

```python
parser.add_argument("--image_size",     type=int,   help="모델 입력 이미지 크기 설정", default=384)
parser.add_argument("--batch_size",     type=int,   help="배치 사이즈 설정", default=32)
parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/model_name.h5')
parser.add_argument("--input_dir", type=str,   help="테스트 이미지 디렉토리 설정", default='./inputs/')
parser.add_argument("--output_dir", type=str,   help="테스트 결과 이미지 디렉토리 설정", default='./outputs/')
parser.add_argument("--backbone_model", type=str,   help="EfficientNet 모델 설정", default='B0')
parser.add_argument("--train_dataset",  type=str,   help="학습에 사용할 dataset 설정 coco or voc", default='voc')
```  
아래와 같이 실행할 수 있습니다.  

    python predcit.py --image_size=384 --dataset_dir='./datasets/' --checkpoint_dir='./checkpoints/model_name.h5'  
    --backbone_model=B0 --train_dataset=voc  --input_dir='./inputs/' --output_dir='./outputs/'


![001070 jpg](https://user-images.githubusercontent.com/60956651/110722231-49632f00-8255-11eb-9351-165d9efac7c2.jpg)
![002107 jpg](https://user-images.githubusercontent.com/60956651/110722280-54b65a80-8255-11eb-8005-0ddd88f33082.jpg)  

## Reference
<hr>

1. [https://github.com/qubvel/efficientnet](https://github.com/qubvel/efficientnet)
2. [https://github.com/keras-team/keras-applications](https://github.com/keras-team/keras-applications)
3. [https://github.com/xuannianz/EfficientDet](https://github.com/xuannianz/EfficientDet)
4. [https://github.com/pierluigiferrari/ssd_keras](https://github.com/pierluigiferrari/ssd_keras)














 