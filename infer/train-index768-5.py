"""
格式：直接cid为自带的index位；aid放不下了，通过字典来查，反正就5w个
"""
import faiss, numpy as np, os,torch

# ###########如果是原始特征要先写save
inp_root = r"E:\codes\py39\test-20230416b\logs\mi-test-v2\3_feature768"
npys = []
for name in sorted(list(os.listdir(inp_root))):
    phone = np.load("%s/%s" % (inp_root, name))
    npys.append(phone)
big_npy = np.concatenate(npys, 0)
torch.from_numpy(big_npy).unfold(0,5,1)#间隔1,5个一组，对第0维进行折叠

print(big_npy.shape)  # (6196072, 192)#fp32#4.43G
np.save("infer/big_src_feature_mi.npy", big_npy)

##################train+add
# big_npy=np.load("/bili-coeus/jupyter/jupyterhub-liujing04/vits_ch/inference_f0/big_src_feature_mi.npy")
print(big_npy.shape)
index = faiss.index_factory(256, "IVF512,Flat")  # mi
print("training")
index_ivf = faiss.extract_index_ivf(index)  #
index_ivf.nprobe = 9
index.train(big_npy)
faiss.write_index(index, "infer/trained_IVF512_Flat_mi_baseline_src_feat.index")
print("adding")
index.add(big_npy)
faiss.write_index(index, "infer/added_IVF512_Flat_mi_baseline_src_feat.index")
"""
大小（都是FP32）
big_src_feature 2.95G
    (3098036, 256)
big_emb         4.43G
    (6196072, 192)
big_emb双倍是因为求特征要repeat后再加pitch

"""
