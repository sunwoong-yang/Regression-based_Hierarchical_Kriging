import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

def optimize(self, dv_idx, obj_idx=None, Morm="m", weights=None):  # pymoo: GA, MOGA, scipy.optimize: gradient-based

    class SO(Problem):
        """
        single-objective
        """

        def __init__(self, dv_idx, Morm="m"):
            super().__init__(n_var=len(dv_idx),
                             n_obj=1,
                             n_constr=0,
                             xl=lower_bound,
                             xu=upper_bound)

            self.Morm = Morm

        def _evaluate(self, x, out, *args, **kwargs):
            obj = self.model.predict(x)

            if weights is None:
                if self.Morm == "M":
                    out["F"] = -np.sum(obj)
                elif self.Morm == "m":
                    out["F"] = np.sum(obj)
            else:
                if self.Morm == "M":
                    out["F"] = -np.sum(np.multiply(obj, weights))
                elif self.Morm == "m":
                    out["F"] = np.sum(np.multiply(obj, weights))

    class MO(Problem):  # Bi-objective optimization

        def __init__(self, dv_idx, obj_idx, Morm=["M", "m"]):
            super().__init__(n_var=len(dv_idx),
                             n_obj=len(obj_idx),
                             n_constr=0,
                             xl=lower_bound,
                             xu=upper_bound)

            self.obj_idx = obj_idx
            self.Morm = Morm

        def _evaluate(self, x, out, *args, **kwargs):
            obj = self.model.predict(x)
            F = []
            for idx in range(len(self.obj_idx)):
                # 여기에 m M 읽고 상황맞게 최적화하도록 수정필요
                if Morm[idx] == "M":
                    F.append(-obj[:, idx])
                elif Morm[idx] == "m":
                    F.append(obj[:, idx])

            out["F"] = np.column_stack(F)

    if obj_idx is None:
        SO(dv_idx, Morm="m")
    else:
        MO(dv_idx, obj_idx, Morm=["M", "m"])