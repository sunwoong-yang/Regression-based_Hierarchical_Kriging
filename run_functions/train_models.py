from surrogate_model.HK import HK
import numpy as np

def train_models(X, Y, add_noise=None, pop=30, gen=100, repetition=1, history=False, rand_seed=42, test_x=None, test_y=None):

	if add_noise is None:
		add_noise = [[0, 0.]] # Default noise: add std=0. Gaussian noise to the lowest fidelity output

	np.random.seed(rand_seed)
	# Use different random seed for training HK models for every repetitive training
	HK_train_seed = np.random.randint(0, 100, size=repetition)
	# Use identical random seed for training HK models for every repetitive training
	# HK_train_seed = np.repeat(np.random.randint(0, 100, size=1), repetition) # 모든 iter에서 동일

	IHKs, RHKs = [], []
	if (test_x is not None) and (test_y is not None):
		i_errors = np.zeros((repetition, 3))
		r_errors = np.zeros((repetition, 3))

	for repeat in range(repetition):
		for noise_fidelity, noise_scale in add_noise:
			Y[noise_fidelity] *= np.random.normal(loc=1, scale=noise_scale, size=(len(Y[noise_fidelity]),1))

		IHK = HK(x=X, y=Y, n_pop=[pop] * len(X), n_gen=[gen] * len(X), HKtype="i")
		IHK.fit(history=history, rand_seed=HK_train_seed[repeat])
		RHK = HK(x=X, y=Y, n_pop=[pop] * len(X), n_gen=[gen] * len(X), HKtype="r")
		RHK.fit(history=history, rand_seed=HK_train_seed[repeat])

		IHKs.append(IHK)
		RHKs.append(RHK)

		if (test_x is not None) and (test_y is not None):
			i_errors[repeat] = IHK.cal_error(test_x, test_y)
			r_errors[repeat] = RHK.cal_error(test_x, test_y)
	return IHKs, RHKs, i_errors, r_errors