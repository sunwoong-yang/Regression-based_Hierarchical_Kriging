from surrogate_model.HK import HK
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

lf_x = np.load("lf_x.npy")
hf_x = np.load("hf_x.npy")
lf_cm = np.load("lf_cm.npy")
hf_cm = np.load("hf_cm.npy")
hf_x_vali = np.load("vali_x.npy")
vali_cm = np.load("vali_cm.npy")
AoA_start, AoA_end = -2, 6
x = np.linspace(AoA_start, AoA_end,41)

###########################################################################
x = [lf_x, hf_x]
cm = [lf_cm, hf_cm]

IHK = HK(x=x, y=cm, n_pop=[30] * len(x), n_gen=[100] * len(x), HKtype="i")
IHK.fit(history=True, rand_seed=42)

RHK = HK(x=x, y=cm, n_pop=[30] * len(x), n_gen=[100] * len(x), HKtype="r")
RHK.fit(history=True, rand_seed=42)

###########################################################################

x_test = np.linspace(AoA_start-0.2, AoA_end+0.2, 1001).reshape(-1,1)

pred_i_lf = IHK.predict(x_test, return_std=False, pred_fidelity=0)
pred_r_lf = RHK.predict(x_test, return_std=False, pred_fidelity=0)
pred_i_hf = IHK.predict(x_test, return_std=False)
pred_r_hf = RHK.predict(x_test, return_std=False)

##############################
# x_test = np.linspace(np.min([np.min(lf_x),np.min(hf_x),np.min(hf_x_vali)]),np.max([np.max(lf_x),np.max(hf_x),np.max(hf_x_vali)]),1001)

############# LF only + MF

fig, ax = plt.subplots(dpi=300)
# figsize=(7,5)
# plt.figure(figsize=(10,5))
current_palette = sns.color_palette("Set2")
ax.scatter(x[0], cm[0], edgecolors='k', facecolors=current_palette[0], s=45, label="Low-fidelity data", zorder=4)
ax.scatter(x[1], cm[1], edgecolors='k', facecolors=current_palette[1], s=45, label="High-fidelity data", zorder=5)

ax.plot(x_test, pred_i_lf, c=current_palette[0], ls='--', label = "Low-fidelity (IHK)", lw=2, zorder=0)
ax.plot(x_test, pred_i_hf, c=current_palette[1], ls='--', label = "High-fidelity (IHK)", lw=2, zorder=1)
ax.plot(x_test, pred_r_lf, c=current_palette[0], ls='-', label = "Low-fidelity (RHK)", lw=2, zorder=2)
ax.plot(x_test, pred_r_hf, c=current_palette[1], ls='-', label = "HIgh-fidelity (RHK)", lw=2, zorder=3)

ax.legend(fontsize=15, loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=3, columnspacing=0.4)
ax.set_xlabel("AoA [deg]",fontsize=15)
ax.set_ylabel("$C_m$",fontsize=15)
left_bot = [0,-0.11]
right_top = [2,-0.09]
left_bot = [0,-0.12]
right_top = [5,-0.09]
ax.plot([left_bot[0], left_bot[0]], [right_top[1], left_bot[1]], '--',color='k', lw=2)
ax.plot([right_top[0], right_top[0]], [right_top[1], left_bot[1]], '--',color='k', lw=2)
ax.plot([left_bot[0], right_top[0]], [right_top[1], right_top[1]], '--',color='k', lw=2)
ax.plot([left_bot[0], right_top[0]], [left_bot[1], left_bot[1]], '--',color='k', lw=2)
ax.set_xlim(AoA_start-0.2, AoA_end+0.2)
fig.savefig("1d_rae_a",bbox_inches='tight')
plt.show()

############# IK IHK 비교위한 확대 그림

fig, ax = plt.subplots(dpi=300, figsize=(2.5,5))
# figsize=(3,4)
# fig = plt.figure(figsize=(3,4))
# ax = fig.add_subplot()
for axxx in ['left','right','top','bottom']:
	ax.spines[axxx].set_color('k')
	ax.spines[axxx].set_linestyle('--')
	ax.spines[axxx].set_linewidth(2)

ax.scatter(x[0], cm[0], edgecolors='k', facecolors=current_palette[0], label="Low-fidelity data", s=45, zorder = 6)
ax.scatter(x[1], cm[1], edgecolors='k', facecolors=current_palette[1], label="High-fidelity data", s=45, zorder = 7)

plt.plot(x_test, pred_i_lf, c=current_palette[0], ls='--',label = "Low-fidelity (IHK)", lw=2,zorder = 2)
plt.plot(x_test, pred_i_hf, c=current_palette[1], ls='--',label = "High-fidelity (IHK)", lw=2,zorder = 3)
plt.plot(x_test, pred_r_lf, c=current_palette[0], ls='-',label = "Low-fidelity (RHK)", lw=2,zorder = 4)
plt.plot(x_test, pred_r_hf, c=current_palette[1], ls='-',label = "High-fidelity (RHK)", lw=2,zorder = 5)

ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
# plt.legend(fontsize=15,loc='center left', bbox_to_anchor=(1, 0.5))
# plt.legend(fontsize=15)
plt.xlim(left_bot[0],right_top[0])
plt.ylim(left_bot[1],right_top[1])
fig.savefig("1d_rae_b",bbox_inches='tight')
plt.show()

