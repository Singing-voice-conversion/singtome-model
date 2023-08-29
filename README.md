# Retreival Voice Conversion (RVC)

## How to run?

### Set virtual environment
anaconda 또는 miniconda를 기준으로 환경을 설정  
cuda version 11.7 또는 11.8 을 기준으로 설명

```commandline
conda create -n rvc-environment python=3.8

# CUDA 11.7 을 사용 중인 경우
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
# CUDA 11.8 을 사용 중인 경우
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -r rvc-requirements.txt
```

[HuggingFace-1](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/pretrained_v2)에서 2가지 파일을 다운로드
- `f0D48k.pth`
- `f0G48k.pth`  
위 2가지 파일은 RVC-model/pretrained_v2 내에 이동

[HuggingFace-2](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main)에서 2가지 파일을 다운로드
- `ffmepg.exe`
- `ffprobe.exe`  
위 2가지 파일은 RVC-model의 최상단에 위치하도록 이동 (rvc-train.py와 동일한 경로)



### Setting parameters

@rvc-train.py  
- trainset_dir4 : 학습에 사용될 데이터셋이 존재하는 폴더를 지정
- exp_dir1 : 학습의 실험 이름을 설정


@rvc-infer.py
- sid0 : weights 폴더내에 실험이름으로 지정되어 있는 pth파일을 선택
- input_audio0 : 추론에 사용할 원본 음악 (vocal file)을 선택
- file_index2 : logs/{exp_dir1}/ 내에 존재하는 added_*.pth 파일을 선택

### Training
```commandline
python rvc-train.py
```

### Inference
```commandline
python rvc-infer.py
```

<br/>

## Reference

- [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
