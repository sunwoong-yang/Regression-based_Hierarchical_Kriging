from error_metric.error_metric import *

def cal_error(ground_truth, i_pred, r_pred):

	ground_truth, i_pred, r_pred = ground_truth.reshape(-1), i_pred.reshape(-1), r_pred.reshape(-1)

	i_RMSE = RMSE(ground_truth, i_pred)
	r_RMSE = RMSE(ground_truth, r_pred)

	i_MAE = MAE(ground_truth, i_pred)
	r_MAE = MAE(ground_truth, r_pred)

	i_Rsq = Rsq(ground_truth, i_pred)
	r_Rsq = Rsq(ground_truth, r_pred)

	return (i_RMSE, i_MAE, i_Rsq), (r_RMSE, r_MAE, r_Rsq)