from surrogate_model.HK import HK

def train_models(X, Y, add_noise=False, repetition=1):

	# for repeat in range(repetition):
	if add_noise:
		pass # 여기에 공통된 noise algorithm을 Y에 부여하자
	IHK = HK(x=X, y=Y, n_pop=[100] * len(X), n_gen=[100] * len(X), HKtype="i")
	IHK.fit()
	RHK = HK(x=X, y=Y, n_pop=[100] * len(X), n_gen=[100] * len(X), HKtype="r")
	RHK.fit()

	return IHK, RHK