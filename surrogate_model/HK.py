from surrogate_model.HK_functions import *

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.termination.robust import RobustTermination
from pymoo.termination.xtol import DesignSpaceTermination
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.operators.mutation.pm import PM
from pymoo.core.problem import ElementwiseProblem
import time


class HK:

	###################################
	def __init__(self, x, y, n_pop=None, n_gen=None, HKtype="r"):
		self.t_start = time.time()
		self.x, self.y = x, y
		self.y = [y.reshape(-1) for y in self.y]
		# for each_y in self.y:
		# 	each_y.reshape(-1)
		if n_pop is None:
			n_pop = [100] * len(x)
		if n_gen is None:
			n_gen = [100] * len(x)
		self.pop, self.gen = np.array(n_pop), np.array(n_gen)
		self.total_level = len(x)
		self.current_level = 0
		self.HKtype = HKtype  # Regression: "r" & Interpolation: "i"
		self.total_opt_theta, self.total_R, self.total_invR, self.total_F, self.total_beta, self.total_sigmaSQ, self.total_MLE = [], [], [], [], [], [], []
		if self.HKtype == ("i" or "I"):
			pass
		elif self.HKtype == ("r" or "R"):
			self.total_opt_nugget, self.total_opt_order = [], []
		else:
			print("Invalid HK type")

	###################################
	def fit(self, history=False, to_level=None):
		if to_level is None:  # to_level 입력안되면 모든 fidelity 학습
			to_level = self.total_level - 1

		while self.current_level < to_level + 1:
			if history:
				print("#########  Level %d starts  #########" % (self.current_level))
			t_temp = time.time()
			x, y = self.x[self.current_level], self.y[self.current_level]

			self.opt_bef_action()
			self.GA_results = self.GA_krig(self.current_level)
			self.opt_X = self.GA_results[0]
			self.opt_aft_action(x, y, self.opt_X)

			result = []
			result.append(self.total_F)
			result.append(self.opt_X)

			announce = "   Final generation = %s" % (self.GA_results[2])
			announce += "\n   Optimal theta = %s" % (self.total_opt_theta[self.current_level])
			if self.HKtype == ("r" or "R"):
				announce += "\n   Optimal nugget = %E" % (self.total_opt_nugget[self.current_level])
			announce += "\n   Optimal likelihood = %f" % (self.total_MLE[self.current_level])
			announce += "\n   Optimal R's condition number = %f" % (np.linalg.cond(self.total_R[self.current_level]))
			announce += "\n   Level %d finishes with time %f[s]" % (self.current_level, time.time() - t_temp)

			if history:
				print(announce)

			self.current_level += 1

		if history:
			print("#########  HK total training time = %f[s]  #########\n" % (time.time() - self.t_start))

		return result, self.total_opt_theta  # $%^&

	###################################
	def opt_bef_action(self):  # x,y,current_level,*args):

		self.N_pts = self.x[self.current_level].shape[0]

		if self.current_level == 0:
			F = np.ones(self.N_pts)
		else:  # if current_level != 0 --> F = estimate으로넣기
			F = self.predict(self.x[self.current_level], self.current_level - 1)[0]

		self.total_F.append(F)

	###################################
	def predict(self, X, pred_fidelity=None, surro_dir=None, return_std=True):  # $%^&
		# HF들의 y와 MSE 계산에는 r_vector와 y_pred의 계산만 새로 필요. 나머지는 새로 계산할 필요 없음

		if pred_fidelity is None:
			pred_fidelity = self.total_level - 1

		if surro_dir is not None:  # $%^&
			self.total_opt_theta = surro_dir  # $%^&

		N_pts_test = self.x[pred_fidelity].shape[0]
		R = self.total_R[pred_fidelity]
		invR = self.total_invR[pred_fidelity]

		if self.HKtype == ("i" or "I"):
			temp_X = self.total_opt_theta[pred_fidelity]
		elif self.HKtype == ("r" or "R"):
			temp_X = self.total_opt_theta[pred_fidelity]
			temp_X = np.append(temp_X, self.total_opt_nugget[pred_fidelity])

		r_vector = cal_r_vector(self.x[pred_fidelity], X, temp_X, self.HKtype)
		F = self.total_F[pred_fidelity]  #### self.total_F >>> total_F
		beta = self.total_beta[pred_fidelity]
		sigmaSQ = self.total_sigmaSQ[pred_fidelity]

		if self.HKtype == ("i" or "I"):
			if pred_fidelity == 0:

				y_pred = beta + r_vector.transpose() @ invR @ (self.y[pred_fidelity] - F * beta)
				MSE = []

				for i in range(X.shape[0]):
					MSE.append(sigmaSQ * (1 - r_vector.transpose()[i] @ invR @ r_vector[:, i] + (
							1 - F.transpose() @ invR @ r_vector[:, i]) ** 2 / (F.transpose() @ invR @ F)))

			else:
				y_lf = self.predict(X, pred_fidelity - 1)[0]
				y_pred = beta * y_lf + r_vector.transpose() @ invR @ (self.y[pred_fidelity] - F * beta)

				temp_1 = 1 / (F.transpose() @ invR @ F)
				MSE = []

				for i in range(X.shape[0]):
					temp_2 = r_vector.transpose()[i] @ invR @ r_vector[:, i]
					temp_3 = r_vector.transpose()[i] @ invR @ F

					MSE.append((sigmaSQ * (1 - temp_2 + (temp_3 - y_lf[i]) * (temp_1) * (temp_3 - y_lf[i]))))

		if self.HKtype == ("r" or "R"):
			# nugget 그냥 빼버리면 inv 계산에서 또 수치에러 발생. 이를 완화위해 trick으로 10**-9 fixed nugget 사용
			regression_invR = np.linalg.inv(
				R - self.total_opt_nugget[pred_fidelity] * np.identity(N_pts_test) + 10 ** -9 * np.identity(N_pts_test))
			regression_sigmaSQ = cal_regression_sigmaSQ(N_pts_test, self.y[pred_fidelity], F, beta, R, invR,
			                                            self.total_opt_nugget[pred_fidelity])

			if pred_fidelity == 0:

				y_pred = beta + r_vector.transpose() @ invR @ (self.y[pred_fidelity] - F * beta)
				MSE = []

				for i in range(X.shape[0]):
					MSE.append(regression_sigmaSQ * (1 - r_vector.transpose()[i] @ regression_invR @ r_vector[:, i] + (
							1 - F.transpose() @ regression_invR @ r_vector[:, i]) ** 2 / (
							                                 F.transpose() @ regression_invR @ F)))


			else:

				y_lf = self.predict(X, pred_fidelity - 1)[0]
				y_pred = beta * y_lf + r_vector.transpose() @ invR @ (self.y[pred_fidelity] - F * beta)

				temp_1 = 1 / (F.transpose() @ regression_invR @ F)
				MSE = []

				for i in range(X.shape[0]):
					temp_2 = r_vector.transpose()[i] @ regression_invR @ r_vector[:, i]
					temp_3 = r_vector.transpose()[i] @ regression_invR @ F

					MSE.append((regression_sigmaSQ * (1 - temp_2 + (temp_3 - y_lf[i]) * (temp_1) * (temp_3 - y_lf[i]))))

		MSE = np.array(MSE)
		MSE[MSE < 0] = 0

		if return_std:
			return y_pred, np.sqrt(MSE)
		else:
			return y_pred
	###################################
	def opt_aft_action(self, x, y, opt_X):

		N_pts = self.x[self.current_level].shape[0]
		R = cal_R(x, y, opt_X, self.HKtype)
		invR = np.linalg.inv(R)

		F = self.total_F[self.current_level]

		transF = F.transpose()
		beta = cal_beta(self.y[self.current_level], F, invR, transF)
		sigmaSQ = cal_sigmaSQ(N_pts, self.y[self.current_level], F, beta, invR)
		MLE = cal_MLE(N_pts, sigmaSQ, R)

		if self.HKtype == ("i" or "I"):
			opt_theta = opt_X

		if self.HKtype == ("r" or "R"):
			opt_theta = opt_X[:-1]
			opt_nugget = opt_X[-1]
			self.total_opt_nugget.append(opt_nugget)

		self.total_opt_theta.append(opt_theta)
		self.total_R.append(R)
		self.total_invR.append(invR)
		self.total_beta.append(beta)
		self.total_sigmaSQ.append(sigmaSQ)
		self.total_MLE.append(MLE)

	###################################

	###################################
	# def plot_var(self, x_test, level, function=None):
	# 	plt.style.use('seaborn-ticks')
	#
	# 	total_level = len(self.x)
	# 	y_HK = self.prediction(x_test, level)
	# 	col = ['b', 'r', 'g']
	# 	for i in range(level + 1):
	# 		# plt.scatter(x[i],y[i],marker = 'o',color='red')
	# 		label_ = "Fidelity-level %d data" % i
	# 		plt.scatter(self.x[i], self.y[i], label=label_, color=col[i])
	# 	if function:
	# 		plt.plot(x_test, function(x_test), 'r-', label="True function")
	# 	if self.HKtype == ("r" or "R") and level > 0:
	# 		plt.plot(x_test, y_HK[0], 'k-', label="Regression-based HK")
	# 	elif self.HKtype == ("i" or "I") and level > 0:
	# 		plt.plot(x_test, y_HK[0], 'k-', label="Interpolation-based HK")
	# 	if self.HKtype == ("r" or "R") and level == 0:
	# 		plt.plot(x_test, y_HK[0], 'k-', label="Regression-based Kriging")
	# 	elif self.HKtype == ("i" or "I") and level == 0:
	# 		plt.plot(x_test, y_HK[0], 'k-', label="Interpolation-based Kriging")
	# 	# plt.fill_between(x_test, y_HK[0]-2*y_HK[1], y_HK[0]+2*y_HK[1],
	# 	#               facecolor="red", # The fill color
	# 	#               edgecolor='black',       # The outline color
	# 	#               alpha=0.3,
	# 	#               linestyle='--')          # Transparency of the fill
	#
	# 	plt.legend()
	# 	plt.show()
	#
	# ###################################
	# def accuracy_plot(self, x_real, y_real, level):
	# 	y_pred = self.prediction(x_real, level)[0]
	# 	limit = np.array([np.min([y_pred, y_real]), np.max([y_pred, y_real])])
	# 	plt.plot(limit, limit, c='k')
	# 	plt.xlabel("Real value", fontsize=15)
	# 	plt.ylabel("Predicted value", fontsize=15)
	# 	plt.scatter(y_real, y_pred, color='k')
	# 	plt.legend(fontsize=15)
	# 	plt.show()
	# 	r_squared = self.Rsq(x_real, y_real, level)
	# 	print(f"R_sq: {r_squared:.4f}")

	###################################
	def pred_arbit_theta(self, x_test, current_level, theta, nugget):

		N_pts = self.x[current_level].shape[0]
		R = cal_R(self.x[current_level], theta[current_level], nugget)
		print("cond", current_level, np.linalg.cond(R))
		invR = np.linalg.inv(R)
		r_vector = cal_r_vector(x_test, self.x[current_level], theta[current_level])
		F = self.total_F[current_level]
		transF = F.transpose()
		beta = cal_beta(self.y[current_level], F, invR, transF)
		sigmaSQ = cal_sigmaSQ(N_pts, self.y[current_level], F, beta, invR)
		MLE = cal_MLE(N_pts, sigmaSQ, R)

		if current_level == 0:

			y = beta + r_vector.transpose() @ invR @ (self.y[current_level] - F * beta)
			MSE = []

			for i in range(x_test.shape[0]):
				MSE.append(sigmaSQ * (1 - r_vector.transpose()[i] @ invR @ r_vector[:, i] + (
						1 - F.transpose() @ invR @ r_vector[:, i]) ** 2 / (F.transpose() @ invR @ F)))

			MSE = np.array(MSE)
			MSE[MSE < 0] = 0
			return y, np.sqrt((MSE)), MLE

		else:
			y_lf = pred_arbit_theta(x_test, current_level - 1, theta, nugget)[0]
			y = beta * y_lf + r_vector.transpose() @ invR @ (self.y[current_level] - F * beta)
			temp_1 = 1 / (F.transpose() @ invR @ F)
			MSE = []
			for i in range(x_test.shape[0]):
				temp_2 = r_vector.transpose()[i] @ invR @ r_vector[:, i]
				temp_3 = r_vector.transpose()[i] @ invR @ F
				MSE.append((sigmaSQ * (1 - temp_2 + (temp_3 - y_lf[i]) * (temp_1) * (temp_3 - y_lf[i]))))

			MSE = np.array(MSE)
			MSE[MSE < 0] = 0
			return y, np.sqrt((MSE)), MLE

	###################################
	def GA_krig(self, current_level):
		n_var = self.x[current_level].shape[1]
		pop_size = self.pop[current_level]
		gen_size = self.gen[current_level]
		fixed_gen = 0  # 1이면 무조건 해당 gen_size만큼 GA
		# if fixed_gen == 1 :
		#   gen_size = 300
		# elif fixed_gen == 0:
		#   gen_size = 2000

		# nested class 때문에 아래와 같이 새로 정의
		x, y = self.x[current_level], self.y[current_level]
		HKtype = self.HKtype
		total_F = self.total_F

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

		if HKtype == ("i" or "I"):  # Hyper-parameter : theta only

			class MyProblem(ElementwiseProblem):

				def __init__(self):
					super().__init__(n_var=n_var,
					                 n_obj=1,
					                 n_constr=0,
					                 xl=np.array([-6.] * n_var),
					                 xu=np.array([3.] * n_var),

					                 )

				def _evaluate(self, X, out, *args, **kwargs):
					X = 10 ** X  # theta는 log scale로 최적화
					asdf = GA_cal_kriging(x, y, X, current_level)
					asdf = asdf.astype('float32')
					obj1 = -asdf
					out["F"] = np.column_stack([obj1])

			# class MyDisplay(Display):
			#
			# 	def _do(self, problem, evaluator, algorithm):
			# 		super()._do(problem, evaluator, algorithm)

			problem = MyProblem()

			algorithm = GA(pop_size=pop_size,
			               mutation=PM(prob=0.2),
			               eliminate_duplicates=True
			               )

			if fixed_gen == 1:

				res = minimize(problem,
				               algorithm,
				               ("n_gen", gen_size),
				               #  verbose=True,
				               #  disply = MyDisplay()

				               )
			elif fixed_gen == 0:
				termination = RobustTermination(DesignSpaceTermination(tol=10**-4), period=10)
				res = minimize(problem,
				               algorithm,
				               termination,
				               ("n_gen", gen_size),
				               #  verbose=True,
				               #  disply = MyDisplay()

				               )

			opt = res.X
			opt = 10 ** opt
			# res.algorithm.pop : 마지막 population임 추후 initialization에 사용 가능
			return opt, res.F, res.algorithm.n_gen

		elif HKtype == ("r" or "R"):  # Hyper-parameter : theta, nugget
			class MyProblem(ElementwiseProblem):

				def __init__(self):
					super().__init__(n_var=n_var + 1,  # nugget과 order를 추가로 고려하기에 +2
					                 n_obj=1,
					                 n_constr=0,
					                 xl=np.array([-6.] * n_var + [-12.]),  # cubit spline --> 변수개수 + nugget
					                 xu=np.array([3.] * n_var + [0.]),

					                 )

				def _evaluate(self, X, out, *args, **kwargs):
					X = 10 ** X  # theta랑 nugget은 log scale로 최적화
					asdf = GA_cal_kriging(x, y, X, current_level)
					asdf = asdf.astype('float32')
					obj1 = -asdf
					out["F"] = np.column_stack([obj1])

			# class MyDisplay(Display):
			#
			# 	def _do(self, problem, evaluator, algorithm):
			# 		super()._do(problem, evaluator, algorithm)

			problem = MyProblem()

			algorithm = GA(pop_size=pop_size,
			               mutation=PM(prob=0.2),
			               eliminate_duplicates=True
			               )

			if fixed_gen == 1:

				res = minimize(problem,
				               algorithm,
				               ("n_gen", gen_size),
				               #  verbose=True,
				               #  disply = MyDisplay()

				               )
			elif fixed_gen == 0:
				termination = RobustTermination(DesignSpaceTermination(tol=10**-4), period=10)
				res = minimize(problem,
				               algorithm,
				               termination,
				               ("n_gen", gen_size),
				               #  verbose=True,
				               #  disply = MyDisplay()

				               )

			opt = res.X
			opt = 10 ** opt
			return opt, res.F, res.algorithm.n_gen

	###################################
	def opt_on_surrogate(self, xl, xu, pop, gen, current_level, VALorEI, morM="M"):
		# morM : y값을 최소화면 "m" 최대화면 "M"
		# VALorEI : 함수값 최적화면 "VAL" EI 최적화면 "EI" VFEI 최적화면 "VFEI"
		n_var = self.x[current_level].shape[1]
		pop_size = pop
		gen_size = gen
		total_level = self.total_level
		if VALorEI == "VFEI":  # 변수 하나 늘려야됨: fidelity level
			xl = np.append(xl, 0)
			xu = np.append(xu, total_level - 1)
			n_var += 1

		# nested class 때문에 아래와 같이 외부의 함수를 부르는 함수 생성
		def predict(x_test, pred_fidelity):
			return self.predict(x_test, pred_fidelity)

		def opt_cal_EI(x_test, current_level, morM):
			return self.cal_EI(x_test, current_level, morM)

		def opt_cal_VFEI(x_test, current_level, morM):
			return self.cal_VFEI(x_test, current_level, morM)

		class MyProblem(ElementwiseProblem):

			def __init__(self):
				if VALorEI != "VFEI":  # VFEI가 아닐 때
					super().__init__(n_var=n_var,
					                 n_obj=1,
					                 n_constr=0,
					                 xl=xl,
					                 xu=xu,

					                 )
				else:  # VFEI는 level까지 최적화 변수에 포함되어 특별한 처리가 필요 "https://pymoo.org/customization/mixed_variable_problem.html"
					super().__init__(n_var=n_var,  # 마지막 변수는 fidelity level
					                 n_obj=1,
					                 n_constr=0,
					                 xl=xl,
					                 xu=xu,

					                 )

			def _evaluate(self, X, out, *args, **kwargs):
				X = np.array([X])

				if VALorEI == "VAL" and morM == "m":
					asdf = opt_pred_y_MSE(X, current_level)[0]

				elif VALorEI == "VAL" and morM == "M":
					asdf = -opt_pred_y_MSE(X, current_level)[0]

				elif VALorEI == "EI":  # EI는 항상 maximize
					asdf = -opt_cal_EI(X, current_level, morM)

				elif VALorEI == "VFEI":  # VFEI는 항상 maximize
					asdf = -opt_cal_VFEI(X[0, :-1], X[0, -1], morM)

				obj1 = asdf.astype('float32')
				out["F"] = np.column_stack([obj1])

		problem = MyProblem()

		if VALorEI != "VFEI":  # VFEI가 아닐 때
			algorithm = GA(pop_size=pop_size,
			               mutation=PM(prob=0.2),
			               eliminate_duplicates=True
			               )

		else:  # VFEI는 level까지 최적화 변수에 포함되어 특별한 처리가 필요 "https://pymoo.org/customization/mixed_variable_problem.html"
			mask = ["real"] * (n_var - 1) + ["int"]  # n_var 개수만큼의 real dv와 1개의 int dv (fidelity level)

			sampling = MixedVariableSampling(mask, {
				"real": get_sampling("real_random"),
				"int": get_sampling("int_random")
			})

			crossover = MixedVariableCrossover(mask, {
				"real": get_crossover("real_sbx", prob=1.0, eta=3.0),
				"int": get_crossover("int_sbx", prob=1.0, eta=3.0)
			})

			mutation = MixedVariableMutation(mask, {
				"real": get_mutation("real_pm", eta=3.0),
				"int": get_mutation("int_pm", eta=3.0)
			})

			algorithm = GA(pop_size=pop_size,
			               sampling=sampling,
			               crossover=crossover,
			               mutation=mutation,
			               # mutation=get_mutation("real_pm", prob=0.2),
			               eliminate_duplicates=True
			               )
		if VALorEI != "VFEI":  # VFEI가 아닐 때
			termination = RobustTermination(DesignSpaceTermination(tol=10**-4), period=10)

			res = minimize(problem,
			               algorithm,
			               termination,
			               ("n_gen", gen_size)
			               #  verbose=True,
			               )
		else:
			res = minimize(problem,
			               algorithm,
			               ("n_gen", gen_size)
			               #  verbose=True,

			               )

		opt = res.X

		if VALorEI == "VAL" and morM == "m":
			return opt, res.F
		elif VALorEI == "VAL" and morM == "M":
			return opt, -res.F
		elif VALorEI == "EI":
			return opt, -res.F
		elif VALorEI == "VFEI":
			return opt[-1], opt[:-1], -res.F  # opt fidelity level, opt dv, opt value

	###################################
	def cal_EI(self, x_test, current_level, morM):
		# morM : 함수를 최소화면 "m" 최대화면 "M"
		x_test = x_test.reshape(-1, self.x[current_level].shape[1])

		def I(x_test, current_level, morM):
			if morM == "m":
				return np.min(self.y[current_level]) - self.predict(x_test, current_level)[0]  # ymin - ypred
			elif morM == "M":
				return self.predict(x_test, current_level)[0] - np.max(self.y[current_level])  # ypred - ymax

		I = I(x_test, current_level, morM)
		s = self.predict(x_test, current_level)[1]
		EI = np.zeros(1)

		for enu, x in enumerate(s):  # s > 0
			if x > 0:
				EI[enu] = I[enu] * scipy.stats.norm.cdf(I[enu] / s[enu]) + s[enu] * scipy.stats.norm.pdf(
					I[enu] / s[enu])

		return EI

	###################################
	def cal_VFEI(self, x_test, current_level, morM):
		# morM : 함수를 최소화면 "m" 최대화면 "M"
		# level과 dv를 동시에 고려한 최적화가 필요한데 이때 아래 참고
		# https://pymoo.org/customization/mixed_variable_problem.html
		x_test = x_test.reshape(-1, self.x[current_level].shape[1])

		def I(x_test, current_level, morM):
			if morM == "m":
				return np.min(self.y[current_level]) - self.predict(x_test, current_level)[0]  # ymin - ypred
			elif morM == "M":
				return self.predict(x_test, current_level)[0] - np.max(self.y[current_level])  # ypred - ymax

		I = I(x_test, self.total_level - 1, morM)  # VF_EI는 Highest-fidelity 기준이여서 self.total_level-1을 넣어야
		s = self.predict(x_test, current_level)[1]

		for level in range(self.total_level - current_level - 1):
			s *= self.total_beta[-level - 1 - 1]

		s = np.sqrt(np.square(s))
		EI = np.zeros(1)

		for enu, x in enumerate(s):  # s > 0
			if x > 0:
				EI[enu] = I[enu] * scipy.stats.norm.cdf(I[enu] / s[enu]) + s[enu] * scipy.stats.norm.pdf(
					I[enu] / s[enu])

		return EI

	###################################
	def Rsq(self, x, y_real, level):
		y_pred = self.pred_y_MSE(x, level)[0]
		correlation_matrix = np.corrcoef(y_pred, y_real)
		correlation_xy = correlation_matrix[0, 1]
		return correlation_xy ** 2

	###################################
	def RMSE(self, x, y_real, level):
		y_pred = self.predict(x, level)[0]
		ans = np.sqrt((np.sum((y_pred - y_real) ** 2)) / y_real.shape[0])
		return ans

	###################################
	def MAE(self, x, y_real, level):
		y_pred = self.predict(x, level)[0]
		ans = np.sum(np.abs(y_pred - y_real)) / y_real.shape[0]
		return ans
