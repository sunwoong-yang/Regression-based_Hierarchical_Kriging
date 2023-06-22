from surrogate_model.HK_functions import *

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.util.termination.x_tol import DesignSpaceToleranceTermination
from pymoo.optimize import minimize
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.core.problem import ElementwiseProblem
import pickle


class HK:
	def __init__(self, XList, YList, qDataFusion):
		self.__XList = XList
		self.__YList = YList
		self.__qDataFusion = qDataFusion

		self.__nLevel = len(XList)

		# HK Variables
		self.__total_opt_theta = []
		self.__total_opt_nugget = []
		self.__total_F = []
		self.__total_R = []
		self.__total_invR = []
		self.__total_beta = []
		self.__total_sigmaSQ = []
		self.__total_MLE = []

		# self.__fdir = fdir
		# self.__fpath_hkVaraible = os.path.join(fdir, "HKVariable.hk")

		# Model Parameter
		self.__hkVariableList = []

		return

	def regression(self):
		nobj = len(self.__YList[0][0])
		for iobj in range(0, nobj, 1):
			print(f"{iobj + 1}th Y Start")
			# Initialize variables for every Y
			self.__total_opt_theta = []
			self.__total_opt_nugget = []
			self.__total_F = []
			self.__total_R = []
			self.__total_invR = []
			self.__total_beta = []
			self.__total_sigmaSQ = []
			self.__total_MLE = []

			for i in range(0, self.__nLevel, 1):
				x = self.__XList[i]
				y = self.__YList[i][:, iobj]

				print(f"   Level{i} Start")
				self.__opt_bef_action(i, x, iobj)

				#                 print(f"Finish opt_bef_action")
				GA_results = self.__GA_krig(i, x, y)

				#                 print(f"Finish GA_krig")
				opt_X = GA_results[0]
				self.__opt_aft_action(i, opt_X, x, y)
			#                 print(f"Finish opt_aft_action")

			args = [self.__total_opt_theta, self.__total_opt_nugget, self.__total_F, self.__total_R,
			        self.__total_invR, self.__total_beta, self.__total_sigmaSQ, self.__total_MLE]

			# Set Parameter
			hkVariable = HKVariable()
			hkVariable.setParameter(args)

			self.__hkVariableList.append(hkVariable)

		return

	def saveParameter(self, fdir):
		for i in range(0, len(self.__hkVariableList), 1):
			f = open(os.path.join(fdir, f"HKVariable{i + 1}.hk"), "wb")
			pickle.dump(self.__hkVariableList[i], f)
			f.close()

		return

	def loadParameter(self, fdir, iobj):
		f = open(os.path.join(fdir, f"HKVariable{iobj + 1}.hk"), "rb")
		hkVariable = pickle.load(f)
		f.close()

		self.__total_opt_theta, self.__total_opt_nugget, self.__total_F, self.__total_R, \
			self.__total_invR, self.__total_beta, self.__total_sigmaSQ, self.__total_MLE = hkVariable.getParameter()

		return

	def prediction(self, x_test, test_level, iobj):

		#         HKtype = self.__qDataFusion["HKtype"]

		HKtype = "Regression"
		N_pts_test = self.__XList[test_level].shape[0]
		R = self.__total_R[test_level]
		invR = self.__total_invR[test_level]

		if HKtype == "Interpolation":
			temp_X = self.__total_opt_theta[test_level]
		else:  # HKtype == "Regression":
			temp_X = self.__total_opt_theta[test_level]
			temp_X = np.append(temp_X, self.__total_opt_nugget[test_level])

		r_vector = cal_r_vector(self.__XList[test_level], x_test, temp_X, HKtype)
		F = self.__total_F[test_level]
		beta = self.__total_beta[test_level]
		sigmaSQ = self.__total_sigmaSQ[test_level]

		if HKtype == "Interpolation":
			if test_level == 0:
				y_pred = beta + r_vector.transpose() @ invR @ (self.__YList[test_level][:, iobj] - F * beta)
				MSE = []

				for i in range(x_test.shape[0]):
					MSE.append(sigmaSQ * (1 - r_vector.transpose()[i] @ invR @ r_vector[:, i] + (
							1 - F.transpose() @ invR @ r_vector[:, i]) ** 2 / (F.transpose() @ invR @ F)))

			else:
				y_lf = self.prediction(x_test, test_level - 1, iobj)[0]
				y_pred = beta * y_lf + r_vector.transpose() @ invR @ (self.__YList[test_level][:, iobj] - F * beta)

				temp_1 = 1 / (F.transpose() @ invR @ F)
				MSE = []

				for i in range(x_test.shape[0]):
					temp_2 = r_vector.transpose()[i] @ invR @ r_vector[:, i]
					temp_3 = r_vector.transpose()[i] @ invR @ F

					MSE.append((sigmaSQ * (1 - temp_2 + (temp_3 - y_lf[i]) * (temp_1) * (temp_3 - y_lf[i]))))

		else:  # HKtype == "Regression"
			# nugget 그냥 빼버리면 inv 계산에서 또 수치에러 발생. 이를 완화위해 trick으로 10**-9 fixed nugget 사용
			regression_invR = np.linalg.inv(
				R - self.__total_opt_nugget[test_level] * np.identity(N_pts_test) + 10 ** -9 * np.identity(
					N_pts_test))
			regression_sigmaSQ = cal_regression_sigmaSQ(N_pts_test, self.__YList[test_level][:, iobj], F, beta, R,
			                                            invR, self.__total_opt_nugget[test_level])

			if test_level == 0:
				y_pred = beta + r_vector.transpose() @ invR @ (self.__YList[test_level][:, iobj] - F * beta)
				MSE = []

				for i in range(x_test.shape[0]):
					MSE.append(regression_sigmaSQ * (
							1 - r_vector.transpose()[i] @ regression_invR @ r_vector[:, i] + (
							1 - F.transpose() @ regression_invR @ r_vector[:, i]) ** 2 / (
									F.transpose() @ regression_invR @ F)))

			else:
				y_lf = self.prediction(x_test, test_level - 1, iobj)[0]
				y_pred = beta * y_lf + r_vector.transpose() @ invR @ (self.__YList[test_level][:, iobj] - F * beta)

				temp_1 = 1 / (F.transpose() @ regression_invR @ F)
				MSE = []

				for i in range(x_test.shape[0]):
					temp_2 = r_vector.transpose()[i] @ regression_invR @ r_vector[:, i]
					temp_3 = r_vector.transpose()[i] @ regression_invR @ F

					MSE.append((regression_sigmaSQ * (
							1 - temp_2 + (temp_3 - y_lf[i]) * (temp_1) * (temp_3 - y_lf[i]))))

		MSE = np.array(MSE)
		MSE[MSE < 0] = 0

		return y_pred, np.sqrt(MSE)

	def __opt_aft_action(self, iLevel, opt_X, x, y):
		#         HKType = self.__qDataFusion["HKType"]
		HKType = "Regression"

		N_pts = x[iLevel].shape[0]
		R = cal_R(x, y, opt_X, HKType)
		invR = np.linalg.inv(R)

		F = self.__total_F[iLevel]
		transF = F.transpose()

		beta = cal_beta(y, F, invR, transF)
		sigmaSQ = cal_sigmaSQ(N_pts, y, F, beta, invR)
		MLE = cal_MLE(N_pts, sigmaSQ, R)

		if HKType == "Interpolation":
			opt_theta = opt_X

		else:  # self.q["HKtype"] == "Regression":
			opt_theta = opt_X[:-1]
			opt_nugget = opt_X[-1]
			self.__total_opt_nugget.append(opt_nugget)

		self.__total_opt_theta.append(opt_theta)
		self.__total_R.append(R)
		self.__total_invR.append(invR)
		self.__total_beta.append(beta)
		self.__total_sigmaSQ.append(sigmaSQ)
		self.__total_MLE.append(MLE)

		return

	def __opt_bef_action(self, iLevel, x, iobj):

		if iLevel == 0:
			N_pts = x.shape[0]
			F = np.ones(N_pts)
		else:  # if current_level != 0 --> F = estimate으로넣기
			F = self.prediction(x, iLevel - 1, iobj)[0]

		self.__total_F.append(F)

		return

	def __GA_krig(self, iLevel, x, y):
		qLevel = self.__qDataFusion[f"Level{iLevel + 1}"]
		n_var = x.shape[1]
		pop_size = qLevel["npop"]
		gen_size = qLevel["nga"]

		#         HKtype = self.__qDataFusion["HKType"]
		HKtype = "Regression"
		total_F = self.__total_F

		def GA_cal_kriging(x, y, X, current_level):
			N_pts = x.shape[0]
			R = cal_R(x, y, X, HKtype)
			invR = np.linalg.inv(R)
			F = total_F[current_level]
			transF = F.transpose()

			beta = cal_beta(y, F, invR, transF)
			sigmaSQ = cal_sigmaSQ(N_pts, y, F, beta, invR)
			MLE = cal_MLE(N_pts, sigmaSQ, R)

			return MLE

		if HKtype == "Interpolation":  # Hyper-parameter : theta only

			class MyProblem(ElementwiseProblem):
				def __init__(self):
					super().__init__(n_var=n_var,
					                 n_obj=1,
					                 n_constr=0,
					                 xl=np.array([-6.] * n_var),
					                 xu=np.array([3.] * n_var))

				def _evaluate(self, X, out, *args, **kwargs):
					X = 10 ** X  # theta는 log scale로 최적화
					asdf = GA_cal_kriging(x, y, X, iLevel)
					asdf = asdf.astype('float32')
					obj1 = -asdf
					out["F"] = np.column_stack([obj1])

			# class MyDisplay(Display):
			#     def _do(self, problem, evaluator, algorithm):
			#         super()._do(problem, evaluator, algorithm)

			problem = MyProblem()

			algorithm = GA(pop_size=pop_size,
			               mutation=get_mutation("real_pm", prob=0.2),
			               eliminate_duplicates=True)

			termination = DesignSpaceToleranceTermination(tol=10 ** -4, n_last=40)
			res = minimize(problem, algorithm, termination, ("n_gen", gen_size),
			               #  verbose=True, #  disply = MyDisplay()
			               )

			opt = res.X
			opt = 10 ** opt
			# res.algorithm.pop : 마지막 population임 추후 initialization에 사용 가능
			return opt, res.F, res.algorithm.n_gen

		else:  # HKtype == "Regression": Hyper-parameter : theta, nugget
			class MyProblem(ElementwiseProblem):
				def __init__(self):
					super().__init__(n_var=n_var + 1,  # nugget과 order를 추가로 고려하기에 +2
					                 n_obj=1,
					                 n_constr=0,
					                 xl=np.array([-6.] * n_var + [-12.]),  # cubit spline --> 변수개수 + nugget
					                 xu=np.array([3.] * n_var + [0.]))

				def _evaluate(self, X, out, *args, **kwargs):
					X = 10 ** X  # theta랑 nugget은 log scale로 최적화
					asdf = GA_cal_kriging(x, y, X, iLevel)
					asdf = asdf.astype('float32')
					obj1 = -asdf
					out["F"] = np.column_stack([obj1])

			# class MyDisplay(Display):
			#     def _do(self, problem, evaluator, algorithm):
			#         super()._do(problem, evaluator, algorithm)

			problem = MyProblem()

			algorithm = GA(pop_size=pop_size,
			               mutation=get_mutation("real_pm", prob=0.2),
			               eliminate_duplicates=True)

			termination = DesignSpaceToleranceTermination(tol=10 ** -4, n_last=40)
			res = minimize(problem, algorithm, termination, ("n_gen", gen_size),
			               #  verbose=True, #  disply = MyDisplay()
			               )

			opt = res.X
			opt = 10 ** opt

			return opt, res.F, res.algorithm.n_gen


# In[70]:


from xml.dom.minidom import parseString
from xml.etree import ElementTree

import os
import copy
import numpy as np


class DictExchange:
	def __init__(self):
		return

	def dict_to_xml(self, q, fdir, fname):
		"""
		:param q: Dictionary
		:param fdir: file path
		:param fname: file name
		:return:
		"""

		fpath = os.path.join(fdir, fname)

		xml = self._dict_to_xml(q)
		xml = ElementTree.tostring(xml)
		dom = (parseString(xml)).toprettyxml()

		xmlfile = open(fpath, "w")
		xmlfile.write(dom)
		xmlfile.close()
		return

	def _dict_to_xml(self, q, parent_node=None):
		"""
		:param q: dictionary
		:param parent_node: (None : root, Else : recursive parent)
		:return:
		"""

		def add_node(key, value, parent):
			node = ElementTree.SubElement(parent, key)

			if isinstance(value, list):
				for i in range(0, len(value), 1):
					child = ElementTree.SubElement(node, 'param')
					child.text = str(value[i])
			else:
				node.text = str(value)

			return node

		if parent_node is None:  # root
			node = ElementTree.Element("Items")
		else:
			node = parent_node

		""" Recursive Loop """

		for key, value in q.items():
			if isinstance(value, dict):
				if (key[0] == "F") or (key[0] == "P"):
					_type = key[0:6].upper()
				elif key[0] == "B":
					qb = q["Body"]
					if ("Nose" in qb) or ("Centr" in qb) or ("Aft" in qb):
						_type = "AXIBOD1"
						for _key, _value in qb.items():
							if "ENOSE" in _value:
								_type = "ELLBOD1"
								break
							elif "ECENTR" in _value:
								_type = "ELLBOD1"
								break
							elif "EAFT" in _value:
								_type = "ELLBOD1"
								break
					else:
						if "R" in qb:
							_type = "AXIBOD2"
						else:
							_type = "ELLBOD2"
				elif (key == "Nose"):
					if ("ENOSE" not in value):
						_type = "AXINOSE"
					else:
						_type = "ELLNOSE"
				elif (key == "Centr"):
					if ("ECENTR" not in value):
						_type = "AXICENTR"
					else:
						_type = "ELLCENTR"
				elif (key == "Aft"):
					if ("EAFT" not in value):
						_type = "AXIAFT"
					else:
						_type = "ELLAFT"
				else:
					_type = key.upper()

				_attrib = {"Type": _type, "Name": key}
				child = ElementTree.SubElement(node, "Item", attrib=_attrib)

				self._dict_to_xml(value, child)
			else:
				add_node(key, value, node)

		return node

	def xml_to_dict(self, fdir, fname):
		fpath = os.path.join(fdir, fname)
		doc = ElementTree.parse(fpath)
		root = doc.getroot()

		q = self._parse(root)

		return q

	def _parse(self, item):
		numlist = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-"]
		q = {}
		for child in item:
			if "Type" in child.attrib:  # Dictionary
				_name = child.attrib["Name"]
				_q = self._parse(child)
				q[_name] = _q
			else:
				if len(list(child)) > 0:
					_list = []
					for _child in child:  # List
						if _child.text == "None":
							_list.append(None)
						else:
							if _child.text[0] in numlist:
								if "." in _child.text:
									_list.append(float(_child.text))
								elif "e" in _child.text:
									_list.append(float(_child.text))
								else:
									_list.append(int(_child.text))
							else:
								_list.append(_child.text)
					q[child.tag] = _list
				else:
					if child.text[0] in numlist:  # number
						if "." in child.text:
							q[child.tag] = float(child.text)
						else:
							q[child.tag] = int(child.text)
					else:  # string
						if child.text == "None":
							q[child.tag] = None
						else:
							q[child.tag] = child.text
		return q

	def dv_to_dict(self, q_ori, dvName, dv):
		numlist = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-"]

		if np.ndim(dv) == 1:
			dv = np.resize(dv, (1, len(dv)))

		nsamples = len(dv)
		ndv = len(dvName)

		params = [None] * nsamples
		for i in range(0, nsamples, 1):
			q = copy.deepcopy(q_ori)

			for j in range(0, ndv, 1):
				str = dvName[j].split("_")
				if len(str) == 2:  # Finset1_XLE
					q["Model"][str[0]][str[1]] = dv[i, j]
				else:
					if str[-1][0] in numlist:  # Finset1_CHORD_1
						q["Model"][str[0]][str[1]][int(str[2]) - 1] = dv[i, j]
					else:
						q["Model"][str[0]][str[1]][str[2]] = dv[i, j]

			params[i] = q

		return params


# In[71]:


import pandas as pd
import numpy as np
import torch
from sklearn.metrics import r2_score


def read_dat(filename="DOEset1.csv"):
	with open(filename) as file_:
		file = pd.read_csv(file_)
		file = np.array(file.values)
		inp_dataset = file[:, :7]
		out_dataset = file[:, 7:]

	return inp_dataset, out_dataset


class DataFusionMethodFactory:
	def __init__(self):
		return

	def getDataFusion(self, method, XList, YList, qDdataFusion):
		if method == "HK":
			#             from DataFusionNew.Method.HK.HK import HK
			return HK(XList, YList, qDdataFusion[method])
		else:
			#             from DataFusionNew.Method.MFDNN.MFDNN import MFDNN
			return MFDNN(XList, YList, qDdataFusion[method])


import matplotlib.pyplot as plt

out_coef = ["Cx", "Cy", "Cz", "Cl", "Cm", "Cn"]


def Comparison_plot(model, X, true_y, out_idx=0, NLevel=1, color='k', title=None, label="", saveDir=""):
	NLevel -= 1  # HK모델의 level은 0부터 시작함. MFDNN은 1부터 시작함
	YPredict = model.prediction(X, NLevel, out_idx)[0]
	r2sq = r2_score(true_y[:, out_index], YPredict)
	plt.scatter(true_y[:, out_index], YPredict, edgecolor='k', facecolors=color, s=70,
	            label=label + f"$R^2$={r2sq:.3f}")
	min_ = np.min(np.array([np.min(true_y[:, out_index]), np.min(YPredict)]))
	max_ = np.max(np.array([np.max(true_y[:, out_index]), np.max(YPredict)]))
	plt.plot([min_, max_], [min_, max_], ls='--', c='r', zorder=1, lw=5)
	plt.title(f"{title}: {out_coef[out_index]}", fontsize=20)
	plt.xlabel("Real values", fontsize=20)
	plt.ylabel("Predicted values", fontsize=20)
	if label is not None:
		plt.legend(fontsize=20, frameon=False)
	#     if out_index == 3:
	#         plt.xlim(-0.04,0.04)
	#         plt.ylim(-0.04,0.04)
	return plt


# In[72]:


LF_train_pts = 100
HF_train_pts = 50

Fake2_LF_inp, Fake2_LF_out = read_dat("./HK_fake2_50/Fake2_LF.csv")
Fake2_LF_inp = np.delete(Fake2_LF_inp, obj=1, axis=1)
LF_test_idx = np.random.choice(Fake2_LF_out.shape[0], 1053 - LF_train_pts, replace=False)
LF_test_mask = np.ones(Fake2_LF_out.shape[0], dtype=bool)
LF_test_mask[LF_test_idx] = False

Fake2_HF_inp, Fake2_HF_out = read_dat("./HK_fake2_50/Fake2_HF.csv")
Fake2_HF_inp = np.delete(Fake2_HF_inp, obj=1, axis=1)
test_idx = np.random.choice(Fake2_HF_out.shape[0], 500 - HF_train_pts, replace=False)
test_mask = np.ones(Fake2_HF_out.shape[0], dtype=bool)
test_mask[test_idx] = False

Fake2_LF_inp_test, Fake2_LF_inp_train = Fake2_LF_inp[~LF_test_mask], Fake2_LF_inp[LF_test_mask]
Fake2_LF_out_test, Fake2_LF_out_train = Fake2_LF_out[~LF_test_mask], Fake2_LF_out[LF_test_mask]

Fake2_HF_inp_test, Fake2_HF_inp_train = Fake2_HF_inp[~test_mask], Fake2_HF_inp[test_mask]
Fake2_HF_out_test, Fake2_HF_out_train = Fake2_HF_out[~test_mask], Fake2_HF_out[test_mask]

import time

time_ = time.time()
# for HF_pts in [100,200,300,400,500]:
# for HF_pts in [25,50,75]:
for HF_pts in [HF_train_pts]:
	HF_idx = np.random.choice(Fake2_HF_out_train.shape[0], HF_pts, replace=False)
	XList_fake2 = [Fake2_LF_inp_train, Fake2_HF_inp_train[HF_idx]]
	YList_fake2 = [Fake2_LF_out_train, Fake2_HF_out_train[HF_idx]]

	saveDir = f"HK_fake2_{HF_pts}"
	qDataFusion_fake2 = DictExchange().xml_to_dict(saveDir, "DataFusion_params_fake.xml")
	method_fake2 = ""
	for key, value in qDataFusion_fake2.items():
		method_fake2 = key
		break

	# Regression
	factory_fake2 = DataFusionMethodFactory()
	dataFusion_fake2 = factory_fake2.getDataFusion(method_fake2, XList_fake2, YList_fake2, qDataFusion_fake2)
	dataFusion_fake2.regression()
	dataFusion_fake2.saveParameter(saveDir)
print(time.time() - time_)

# In[74]:


for HF_pts in [HF_train_pts]:

	y_at_LF = []

	for out_index in range(0, 6):
		saveDir = f"HK_fake2_{HF_pts}"
		qDataFusion_fake2 = DictExchange().xml_to_dict(saveDir, "DataFusion_params_fake.xml")
		method_fake2 = ""
		for key, value in qDataFusion_fake2.items():
			method_fake2 = key
			break
		factory_fake2 = DataFusionMethodFactory()
		dataFusion_fake2 = factory_fake2.getDataFusion(method_fake2, XList_fake2, YList_fake2, qDataFusion_fake2)
		dataFusion_fake2.loadParameter(saveDir, out_index)

		Comparison_plot(model=dataFusion_fake2, X=Fake2_HF_inp_test[:], true_y=Fake2_HF_out_test[:], out_idx=out_index,
		                NLevel=1, color='y', label='LF: ', title=f"HK model", saveDir=saveDir)
		Comparison_plot(model=dataFusion_fake2, X=Fake2_HF_inp_test[:], true_y=Fake2_HF_out_test[:], out_idx=out_index,
		                NLevel=2, label='MF: ', title=f"HK model", saveDir=saveDir)
		plt.show()

		y_at_LF.append(dataFusion_fake2.prediction(x_test=XList_fake2[0], test_level=1, iobj=out_index)[0])
	y_at_LF = np.array(y_at_LF).T
	csv_data = np.concatenate((XList_fake2[0], y_at_LF), axis=1)
	csv_data = np.insert(csv_data, obj=1, values=0, axis=1)
	csv_data = pd.DataFrame(csv_data)
	csv_data.to_csv(saveDir + f"/fake2_{HF_pts}.csv", index=False, header=False)

