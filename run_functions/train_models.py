from surrogate_model.HK import HK

def train_models(X, Y, add_noise=False, pop=100, gen=100, repetition=1, history=False):

	# for repeat in range(repetition):
	if add_noise:
		pass # 여기에 공통된 noise algorithm을 Y에 부여하자
	IHK = HK(x=X, y=Y, n_pop=[pop] * len(X), n_gen=[gen] * len(X), HKtype="i")
	IHK.fit(history=history)
	RHK = HK(x=X, y=Y, n_pop=[pop] * len(X), n_gen=[gen] * len(X), HKtype="r")
	RHK.fit(history=history)

	return IHK, RHK