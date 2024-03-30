<div align="center">

# singtome: AI models

*introducing the AI models used in the SINGTOME project.*

[![Static Badge](https://img.shields.io/badge/language-english-red)](./README.md) [![Static Badge](https://img.shields.io/badge/language-korean-blue)](./README-KR.md) [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FSinging-voice-conversion%2Fsingtome-model&count_bg=%23E3E30F&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

</div>

<br>

Hello, and thank you for visiting. This GitHub repository contains the implementation code for the model used in the singtome project. If you're interested in learning more about the singtome project, please refer to [this link](#). The project utilizes the RVC (Retrieval Voice Conversion) model to implement vocal transformation features in music. You can find the core implementation of the model [here](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebU). For more detailed information on how to use the RVC model and its principles, please consult the linked RVC repository.

<br>

<div align="center">

<h3> Model part Team members </h3>

| Profile | Name | GitHub | Role |
| :---: | :---: | :---: | :---: |
| <img src="./img/moon.jpg" height="120px"> | Jongmoon Ryu <br> **moon**| <a href="https://github.com/Orca0917"> <div style="display: flex; align-items: center;"> <img src="./img/github.png" height="20px"> &nbsp; Orca0917 </div> </a> | Model Pipeline and Architecture Design <br> Creating a Docker Image for RVC Model Training & Inference <br> Managing and Operating AWS SageMaker |
| <img src="https://avatars.githubusercontent.com/u/24919880?v=4" height="120px"> | Heechan Chung <br> **anselmo**| <a href="https://github.com/anselmo228"> <div style="display: flex; align-items: center;"> <img src="./img/github.png" height="20px"> &nbsp; anselmo228 </div> </a> | Create a Docker Image for Inferring UVR Models <br> Manage AWS S3 buckets, Lambda, and API gateways <br> Managing singtome project model experiments|

<br>

<h3> Skills </h3>

<img src="./img/skills.png">

</div>

<br>

## 1. Pipeline


In this project, we've implemented a complex process to transform original music tracks into a specific user's voice. To achieve this, we utilized two main models:

1. **UVR (Ultimate Vocal Remover)**: This model separates the background music (MR) and vocals in a music track. Based on a pretrained model, UVR achieves high-quality separation of music and vocals.
2. **RVC (Retrieval Voice Conversion)**: This model is responsible for converting the separated vocals into a specific user's voice. RVC learns the characteristics of a user's voice and applies this to the actual music to create the final output.

<br>

Through these two models, the entire process of converting an original track into the user's voice is implemented as follows:

1. **Music Separation**: The UVR model is used to separate the background music and vocals from the original track.
2. **Voice Conversion**: The separated vocals are input into the RVC model to be converted into a specific user's voice.
3. **Output Generation**: The converted voice is combined with the background music to create the final output.

<br>

This process allows users to experience music converted into their own voice, adding a new dimension to music appreciation. This architecture is designed to clearly understand the complex process involved.

<img src="./img/model_pipeline.png" />

<br>

## 2. Architecture

이 프로젝트는 실제 서비스 환경을 고려하여 설계되었으며, 모든 학습과 추론 작업을 클라우드에서 처리할 수 있도록 AWS (Amazon Web Services)를 선택했습니다. AWS의 강력한 클라우드 인프라를 활용하여, 모델 학습과 추론을 위한 다음과 같은 서비스를 사용하였습니다:

1. **Amazon SageMaker**: 모델 학습을 위해 SageMaker를 사용했습니다. SageMaker는 머신 러닝 모델을 쉽고 빠르게 구축, 학습시키고, 배포할 수 있는 완전 관리형 서비스를 제공합니다.
2. **AWS Lambda**: 학습 및 추론을 위한 트리거(trigger) 기능으로 Lambda 함수를 사용하였습니다. Lambda는 서버를 관리하지 않고도 코드를 실행할 수 있게 해주는 이벤트 기반 컴퓨팅 서비스입니다.

<br>

이 프로젝트의 클라우드 기반 워크플로우는 대략적으로 다음과 같은 과정을 포함합니다:

1. **사용자 요청 수신**: 사용자로부터 원곡 변환 요청을 받습니다.
2. **Lambda 트리거 활성화**: 요청을 처리하기 위해 AWS Lambda 함수가 트리거됩니다.
3. **SageMaker에서 모델 학습 및 추론**: Lambda 함수는 Amazon SageMaker를 호출하여 모델 학습과 추론 작업을 진행합니다.
4. **결과 반환**: 변환된 음악 파일을 사용자에게 반환합니다.

<br>

이 아키텍처는 클라우드의 유연성과 확장성을 최대한 활용하여, 고품질의 사용자 경험을 제공하도록 설계되었습니다. 사용자의 요청에 따라 모델 학습과 추론이 자동으로 진행되며, 이 모든 과정은 클라우드 서비스를 통해 관리됩니다.

<img src="./img/aws_architecture.png" />

<br>

## 3. How does it work?

The model part of the project leverages a range of services provided by AWS, offering high scalability and flexibility in the cloud environment. Here are the key features and processes:

1. **API Gateway**: All requests are received through AWS's API Gateway, which routes each request to the appropriate resource while providing security, monitoring, and usage management.

2. **Amazon SageMaker**: Upon receiving requests, AWS automatically allocates SageMaker instances to perform model training or inference. SageMaker is a fully managed service that facilitates the easy building, training, and deployment of machine learning models.

3. **S3 Bucket Storage**: Trained model parameters or inference results (audio files) are stored in Amazon S3 buckets. Users can access these buckets to download the necessary data.

4. **Spring Boot Backend**: Backend information management (user information, registration details, etc.) is handled by a Spring Boot-based backend. This allows for stable data management separate from the frontend.

5. **Docker and ECR**: To use custom models in Amazon SageMaker, Docker images are uploaded to Amazon Elastic Container Registry (ECR) and then fetched by SageMaker. This approach enhances model management and simplifies modifications to the model's implementation if the input and output formats remain the same.

This structure enables the creation of high-performance, scalable applications through AWS's robust cloud capabilities. Detailed information on additional dependencies and environment settings is provided in the sections below.

<img src="./img/seq_diagram.png" />

<br>

## 4. Environment

All operations were performed within a Docker environment. Therefore, please refer to each `Dockerfile` for the required requirements and base image information. The local GPU and cloud instance information used for training are as follows.

- LOCAL GPU: NVIDIA RTX 4090 x 2
- CLOUD INSTANCE: AWS g4dn.xlarge

### 4.1. RVC-train

[🐳 Goto Dockerfile]()

### 4.2. RVC-inference

[🐳 Goto Dockerfile]()

### 4.3. UVR-inference

[🐳 Goto Dockerfile]()

<br>

### 4.4. Etc.

Here, we introduce how to download the pretrained files required during the training and inference processes. We need two pretrained parameter files, as detailed below. To ensure precise voice processing, it's essential to prepare the `ffmpeg.exe` and `ffprobe.exe` libraries as well.

> [HuggingFace-1](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/pretrained_v2): move files below to RVC-model/pretrained_v2
> - `f0D48k.pth`
> - `f0G48k.pth`

> [HuggingFace-2](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main): move files below to RVC-model/
> - `ffmepg.exe`
> - `ffprobe.exe`

<br/>

### 4.5. Setting parameters


`@rvc-train.py`
- **trainset_dir**: Specifies the folder where the dataset to be used for training is located.
- **exp_dir**: Sets the name of the experiment for the training.

```commandline
python rvc-train.py
```

<br>

`@rvc-infer.py`
- **sid0**: Selects the .pth file specified by the experiment name within the weights folder.
- **input_audio0**: Chooses the original music (vocal file) to be used for inference.
- **file_index2**:  Selects the `added_*.pth` file existing within the `logs/{exp_dir}/` directory.

```commandline
python rvc-infer.py
```

<br/>

## 5. Reference

- [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
