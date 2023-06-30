import numpy as np

def RMSE(ground_truth, prediction):

	# return np.sqrt( np.sum((ground_truth - prediction)**2) / len(ground_truth))
	return np.sqrt(np.sum((ground_truth - prediction) ** 2) / len(ground_truth)) / np.mean(ground_truth)# NRMSE


def MAE(ground_truth, prediction): # Mean Absolute Error
	# MAE = np.sum(np.abs(prediction - ground_truth)) / len(ground_truth)
	MAE = np.sum(np.abs((prediction - ground_truth) / ground_truth)) / len(ground_truth) # MAPE
	return MAE

def Rsq(ground_truth, prediction):

	correlation_matrix = np.corrcoef(prediction, ground_truth)
	correlation_xy = correlation_matrix[0, 1]
	return correlation_xy ** 2
