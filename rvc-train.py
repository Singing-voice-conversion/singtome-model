import os
import shutil
import sys
import traceback
import warnings
import numpy as np
import torch

import logging
from random import shuffle

import faiss
from config import Config
from i18n import I18nAuto
from sklearn.cluster import MiniBatchKMeans


now_dir = os.getcwd()
sys.path.append(now_dir)
logging.getLogger("numba").setLevel(logging.WARNING)

tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % (now_dir), ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/uvr5_pack" % (now_dir), ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)

config = Config()
i18n = I18nAuto()
i18n.print()
# 判断是否有能用来训练和加速推理的N卡
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(
                value in gpu_name.upper()
                for value in [
                    "10",
                    "16",
                    "20",
                    "30",
                    "40",
                    "A2",
                    "A3",
                    "A4",
                    "P4",
                    "A50",
                    "500",
                    "A60",
                    "70",
                    "80",
                    "90",
                    "M4",
                    "T4",
                    "TITAN",
                ]
        ):
            # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            )
if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = i18n("很遗憾您这没有能用的显卡来支持您训练")
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos])


hubert_model = None
weight_root = "weights"
weight_uvr5_root = "uvr5_weights"
index_root = "logs"
names = []

for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)

index_paths = []
for root, dirs, files in os.walk(index_root, topdown=False):
    for name in files:
        if name.endswith(".index") and "trained" not in name:
            index_paths.append("%s/%s" % (root, name))

uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", ""))

sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}

def train1key(
        exp_dir1,
        sr2,
        if_f0_3,
        trainset_dir4,
        spk_id5,
        np7,
        f0method8,
        save_epoch10,
        total_epoch11,
        batch_size12,
        if_save_latest13,
        pretrained_G14,
        pretrained_D15,
        gpus16,
        if_cache_gpu17,
        if_save_every_weights18,
        version19,
):
    model_log_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    preprocess_log_path = "%s/preprocess.log" % model_log_dir
    extract_f0_feature_log_path = "%s/extract_f0_feature.log" % model_log_dir
    gt_wavs_dir = "%s/0_gt_wavs" % model_log_dir
    feature_dir = (
        "%s/3_feature256" % model_log_dir
        if version19 == "v1"
        else "%s/3_feature768" % model_log_dir
    )

    os.makedirs(model_log_dir, exist_ok=True)

    # step 1. process data
    print("\nSTEP1 : Process data")
    print("- running trainset_preprocess_pipleline_print.py")
    open(preprocess_log_path, "w").close()
    cmd = (
            config.python_cmd
            + " trainset_preprocess_pipeline_print.py %s %s %s %s "
            % (trainset_dir4, sr_dict[sr2], np7, model_log_dir)
            + str(config.noparallel)
    )
    os.system(cmd)

    # step 2a. pitch extraction
    print("\nSTEP2a : Pitch extraction")
    print("- running extract_f0_print.py")
    open(extract_f0_feature_log_path, "w")
    if if_f0_3:
        cmd = config.python_cmd + " extract_f0_print.py %s %s %s" % (
            model_log_dir,
            np7,
            f0method8,
        )
        os.system(cmd)

    # step 2b. feature extraction
    print("\nSTEP2b : Feature extraction")
    print("- running extract_feature_print.py")
    gpus = gpus16.split("-")
    leng = len(gpus)
    for idx, n_g in enumerate(gpus):
        cmd = config.python_cmd + " extract_feature_print.py %s %s %s %s %s %s" % (
            config.device,
            leng,
            idx,
            n_g,
            model_log_dir,
            version19,
        )
        os.system(cmd)

    # STEP 3a. Model training
    print("\nSTEP3a : Model training")

    # 生成filelist
    if if_f0_3:
        f0_dir = "%s/2a_f0" % model_log_dir
        f0nsf_dir = "%s/2b-f0nsf" % model_log_dir
        names = (
                set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
                & set([name.split(".")[0] for name in os.listdir(feature_dir)])
                & set([name.split(".")[0] for name in os.listdir(f0_dir)])
                & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % model_log_dir, "w") as f:
        f.write("\n".join(opt))
    if gpus16:
        cmd = (
                config.python_cmd
                + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s"
                % (
                    exp_dir1,
                    sr2,
                    1 if if_f0_3 else 0,
                    batch_size12,
                    gpus16,
                    total_epoch11,
                    save_epoch10,
                    "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                    "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                    1 if if_save_latest13 == i18n("是") else 0,
                    1 if if_cache_gpu17 == i18n("是") else 0,
                    1 if if_save_every_weights18 == i18n("是") else 0,
                    version19,
                )
        )
    else:
        cmd = (
                config.python_cmd
                + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s"
                % (
                    exp_dir1,
                    sr2,
                    1 if if_f0_3 else 0,
                    batch_size12,
                    total_epoch11,
                    save_epoch10,
                    "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                    "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                    1 if if_save_latest13 == i18n("是") else 0,
                    1 if if_cache_gpu17 == i18n("是") else 0,
                    1 if if_save_every_weights18 == i18n("是") else 0,
                    version19,
                )
        )
    print("- running train_nsf_sim_cache_sid_load_pretrain.py")
    os.system(cmd)

    # STEP 3b. Training index
    print("\nSTEP3b : Training index")
    npys = []
    listdir_res = list(os.listdir(feature_dir))
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)

    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]

    if big_npy.shape[0] > 2e5:
        # if(1):
        info = "Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0]
        print(info)
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except:
            info = traceback.format_exc()
            print(info)

    np.save("%s/total_fea.npy" % model_log_dir, big_npy)

    # n_ivf =  big_npy.shape[0] // 39
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    # yield get_info_str("%s,%s" % (big_npy.shape, n_ivf))
    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    # yield get_info_str("training index")
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (model_log_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    # yield get_info_str("adding index")
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i: i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (model_log_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )


# -- Process data
trainset_dir4 = "D:/takoyaki/dataset/uvr/uvr-dataset"
exp_dir1 = "rvc-jungseungwhan"
sr2 = "48k"
np7 = int(np.ceil(config.n_cpu / 1.5))

# -- Feature Extraction
gpus6 = '0'
f0method8 = 'harvest'
if_f0_3 = True
version19 = 'v2'
gpus_rmvpe = '0-0'

# -- One-click training
spk_id5 = 0
save_epoch10 = 10
total_epoch11 = 50
batch_size12 = 8
if_save_latest13 = i18n("否")
pretrained_G14 = 'pretrained_v2/f0G48k.pth'
pretrained_D15 = 'pretrained_v2/f0D48k.pth'
gpus16 = '0'
if_cache_gpu17 = i18n("否")
if_save_every_weights18 = i18n("否")
train1key(
    exp_dir1, sr2, if_f0_3, trainset_dir4, spk_id5, np7, f0method8, save_epoch10,
    total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17,
    if_save_every_weights18, version19
)