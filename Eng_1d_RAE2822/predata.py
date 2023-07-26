from surrogate_model.HK import HK
import numpy as np
import matplotlib.pyplot as plt

# hf_test=np.load("coeff_hf100.npy")
AoA_start, AoA_end = -2, 6
x = np.linspace(AoA_start, AoA_end,41)
hf=np.load("kflow_hf.npy")
lf=np.load("kflow_lf.npy")

plt.scatter(np.linspace(AoA_start, AoA_end,41),lf[:,2])
plt.show()

# HF는 3개 징검다리로 skip
# hf = np.delete(hf,[-1,-2],0) # aoa 16.5과 16은 HF cd 값이 너무 극단적으로증가해서 제외
temp_idx = [6*i+4 for i in range(7)]

temp_idx.remove(28) # 얘는 원래 코드에 있었음! 230723
if not 0 in temp_idx:
  train_idx = [0] + temp_idx
if not 40 in temp_idx:
  train_idx = [40] + temp_idx
# train_idx = [0] + temp_idx + [40] # [29] 값이 outlier니까 포함하게끔 조정하기

total_idx_temp = [i for i in range(hf.shape[0])]
vali_idx = list(set(total_idx_temp) - set(train_idx))

hf_vali = hf[vali_idx]
hf_x_vali = x[vali_idx]
hf = hf[train_idx]
hf_x = x[train_idx]

# 점 개수 너무 많아서 하나씩 skip
# del_idx = [3*i+2 for i in range(int(lf.shape[0]/3)+0)]
# lf = np.delete(lf,del_idx,0)
# lf_x = np.delete(x,del_idx,0)
# lf_idx = [4*i+0 for i in range(11)] +  [4*i+2 for i in range(10)] + [4*i+3 for i in range(10)] #+ [7*i+4 for i in range(6)] + [7*i+5 for i in range(6)] + [7*i+6 for i in range(5)]
# lf_idx.remove(0) # 시작점 제외 (aoa -2)
# lf_idx.remove(40) # 끝점 제외 (aoa 6)
# lf_idx.append(1)
# lf_idx.remove(29) # 얘 값은 너무 튐 (aoa 3.8)
# lf_idx = list(set(total_idx_temp) - set(temp_idx))
lf_idx = [6*i+1 for i in range(7)] + [6*i+2 for i in range(7)] + [6*i+3 for i in range(7)] +  [6*i+5 for i in range(6)]
# lf_idx = [6*i+1 for i in range(7)] + [6*i+3 for i in range(7)] +  [6*i+5 for i in range(6)]
# lf_idx.append(1)
# lf_idx.append(38)
try:
  lf_idx.remove(29)
except:
  pass
try:
  lf_idx.remove(0)
except:
  pass
try:
  lf_idx.remove(40)
except:
  pass

# #임시
# lf_idx.append(0)
# lf_idx.append(40)

lf = lf[lf_idx]
lf_x = x[lf_idx]

hf_cl = hf[:,0]
hf_cd = hf[:,1]
hf_cm = hf[:,2]

# hf_x = np.delete(hf_x, [1,4])
# hf_cm = np.delete(hf_cm, [1,4])

vali_cl = hf_vali[:,0]
vali_cd = hf_vali[:,1]
vali_cm = hf_vali[:,2]

lf_cl = lf[:,0]
lf_cd = lf[:,1]
lf_cm = lf[:,2]

lf_x = np.reshape(lf_x,(-1,1))
hf_x = np.reshape(hf_x,(-1,1))



###########################################################################
x = [lf_x, hf_x]
cm = [lf_cm, hf_cm]

np.save("lf_x.npy", lf_x)
np.save("hf_x.npy", hf_x)
np.save("lf_cm.npy", lf_cm)
np.save("hf_cm.npy", hf_cm)
np.save("vali_x.npy", hf_x_vali)
np.save("vali_cm.npy", vali_cm)