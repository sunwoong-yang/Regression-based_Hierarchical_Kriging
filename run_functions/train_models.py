from surrogate_model.HK import HK

def train_models(X, Y):
	IHK = HK(x=X, y=Y, n_pop=[100, 100, 100], n_gen=[100, 100, 100], HKtype="i")
	IHK.fit()
	RHK = HK(x=X, y=Y, n_pop=[100, 100, 100], n_gen=[100, 100, 100], HKtype="r")
	RHK.fit()

	return IHK, RHK