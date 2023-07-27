from surrogate_model.HK import HK
import numpy as np
import copy
import time
from sklearn.preprocessing import MinMaxScaler

def train_models(X, Y, add_noise=None, pop=30, gen=100, repetition=1, history=False, print_time=True, rand_seed=42, test_x=None, test_y=None):

	x_scaler_list, x_scaled = [], []
	for level, x in enumerate(X):
		if level == 0:
			x_scaler = MinMaxScaler()
			x = x_scaler.fit_transform(x)
		else:
		# x_scaler_list.append(x_scaler)
			x = x_scaler.transform(x)
		x_scaled.append(x)
	X = x_scaled

	test_x = x_scaler.transform(test_x)

	if add_noise is None:
		add_noise = [[0, 0.2, 0.2], [1, 0.1, 0.1]] # Default noise: add std=0.2 Gaussian noise to the lowest fidelity output & std=0.1 to the mid fidelity output

	np.random.seed(rand_seed)
	# Use different random seed for training HK models for every repetitive training
	HK_train_seed = np.random.randint(0, 100, size=repetition)
	# Use identical random seed for training HK models for every repetitive training
	# HK_train_seed = np.repeat(np.random.randint(0, 100, size=1), repetition) # 모든 iter에서 동일

	IHKs, RHKs = [], []
	IHK_time, RHK_time = [], []
	IHK_likeli, RHK_likeli = np.zeros((repetition, 3)), np.zeros((repetition, 3))
	if (test_x is not None) and (test_y is not None):
		i_errors = np.zeros((repetition, 3))
		r_errors = np.zeros((repetition, 3))

	for repeat in range(repetition):
		y_noise = copy.deepcopy(Y)
		for noise_fidelity, noise_multiply, noise_add in add_noise:
			y_noise[noise_fidelity] *= np.random.normal(loc=1, scale=noise_multiply, size=(len(y_noise[noise_fidelity]),1))
			y_noise[noise_fidelity] += np.random.normal(loc=0, scale=noise_add, size=(len(y_noise[noise_fidelity]), 1))

		temp_time = time.time()
		IHK = HK(x=X, y=y_noise, n_pop=[pop] * len(X), n_gen=[gen] * len(X), HKtype="i")
		IHK.fit(history=history, rand_seed=HK_train_seed[repeat])
		IHK_time_ = time.time()-temp_time

		temp_time = time.time()
		RHK = HK(x=X, y=y_noise, n_pop=[pop] * len(X), n_gen=[gen] * len(X), HKtype="r")
		RHK.fit(history=history, rand_seed=HK_train_seed[repeat])
		RHK_time_ = time.time() - temp_time

		IHKs.append(IHK)
		RHKs.append(RHK)
		IHK_time.append(IHK_time_)
		RHK_time.append(RHK_time_)
		IHK_likeli[repeat] = IHK.total_MLE
		RHK_likeli[repeat] = RHK.total_MLE

		if (test_x is not None) and (test_y is not None):
			i_errors[repeat] = IHK.cal_error(test_x, test_y)
			r_errors[repeat] = RHK.cal_error(test_x, test_y)
		print(f"{repeat + 1}th IHK time: {IHK_time_ / 60:.3f} m & error: {i_errors[repeat]}")
		print(f"{repeat + 1}th RHK time: {RHK_time_ / 60:.3f} m & error: {r_errors[repeat]}")
		print(IHK.total_MLE, RHK.total_MLE)
	return IHKs, RHKs, i_errors, r_errors, IHK_likeli, RHK_likeli, np.array(IHK_time), np.array(RHK_time), x_scaler
