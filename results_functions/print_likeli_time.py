import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

IHK_Forrester = np.load("../results_functions/likeli/IHK_Forrester.npy", )
RHK_Forrester = np.load("../results_functions/likeli/RHK_Forrester.npy", )
IHK_Branin = np.load("../results_functions/likeli/IHK_Branin.npy", )
RHK_Branin = np.load("../results_functions/likeli/RHK_Branin.npy", )
IHK_Camel = np.load("../results_functions/likeli/IHK_Camel.npy", )
RHK_Camel = np.load("../results_functions/likeli/RHK_Camel.npy", )
IHK_Func4 = np.load("../results_functions/likeli/IHK_Func4.npy", )
RHK_Func4 = np.load("../results_functions/likeli/RHK_Func4.npy", )
IHK_Func5 = np.load("../results_functions/likeli/IHK_Func5.npy", )
RHK_Func5 = np.load("../results_functions/likeli/RHK_Func5.npy", )
IHK_Func6 = np.load("../results_functions/likeli/IHK_Func6.npy", )
RHK_Func6 = np.load("../results_functions/likeli/RHK_Func6.npy", )

func_name = ["Forrester", "Branin", "Camel", "Func4", "Func5", "Func6"] # 나중에 func 1,2,3,4로 당기기 (기존 func1,2을 안쓰게됨)
type_HK = ["IHK_", "RHK_"]
for level_ in [0,1,2]:
	for func_ in func_name:
		for type_ in type_HK:
			variable_ = eval(type_ + func_)
			print(np.mean(variable_, axis=0)[level_], end =" ")
	print()

print("\nEach row indicates the fidelity level (0~2), each column indicates IHK and RHK of each func")

IHK_Forrester = np.load("../results_functions/time/IHK_Forrester.npy", )
RHK_Forrester = np.load("../results_functions/time/RHK_Forrester.npy", )
IHK_Branin = np.load("../results_functions/time/IHK_Branin.npy", )
RHK_Branin = np.load("../results_functions/time/RHK_Branin.npy", )
IHK_Camel = np.load("../results_functions/time/IHK_Camel.npy", )
RHK_Camel = np.load("../results_functions/time/RHK_Camel.npy", )
IHK_Func4 = np.load("../results_functions/time/IHK_Func4.npy", )
RHK_Func4 = np.load("../results_functions/time/RHK_Func4.npy", )
IHK_Func5 = np.load("../results_functions/time/IHK_Func5.npy", )
RHK_Func5 = np.load("../results_functions/time/RHK_Func5.npy", )
IHK_Func6 = np.load("../results_functions/time/IHK_Func6.npy", )
RHK_Func6 = np.load("../results_functions/time/RHK_Func6.npy", )

func_name = ["Forrester", "Branin", "Camel", "Func4", "Func5", "Func6"]  # 나중에 func 1,2,3,4로 당기기 (기존 func1,2을 안쓰게됨)
type_HK = ["IHK_", "RHK_"]

for func_ in func_name:
	for type_ in type_HK:
		variable_ = eval(type_ + func_)
		print(np.mean(variable_, axis=0), end=" ")

