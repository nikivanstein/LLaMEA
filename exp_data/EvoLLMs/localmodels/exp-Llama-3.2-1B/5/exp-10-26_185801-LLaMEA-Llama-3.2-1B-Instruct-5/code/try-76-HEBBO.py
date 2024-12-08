import numpy as np
from scipy.optimize import minimize

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            res = minimize(lambda x: -func(x), self.search_space, method="SLSQP", bounds=[(-5.0, 5.0)]*self.dim)
            if np.isnan(res.fun) or np.isinf(res.fun):
                raise ValueError("Invalid function value")
            if res.fun < 0 or res.fun > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

class HEBBOWithAdaptiveLineSearch(HEBBO):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.line_search_threshold = 1e-4
        self.line_search_convergence = 1e-6

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            res = minimize(lambda x: -func(x), self.search_space, method="SLSQP", bounds=[(-5.0, 5.0)]*self.dim)
            if np.isnan(res.fun) or np.isinf(res.fun):
                raise ValueError("Invalid function value")
            if res.fun < 0 or res.fun > 1:
                raise ValueError("Function value must be between 0 and 1")
            if np.abs(res.fun - res.fun_last) < self.line_search_threshold:
                line_search_convergence = 1e-6
                while np.abs(res.fun - res.fun_last) > line_search_convergence:
                    if np.isnan(res.fun) or np.isinf(res.fun):
                        raise ValueError("Invalid function value")
                    if res.fun < 0 or res.fun > 1:
                        raise ValueError("Function value must be between 0 and 1")
                    new_individual = res.x + 0.1 * np.random.normal(size=self.dim)
                    res = minimize(lambda x: -func(x), new_individual, method="SLSQP", bounds=[(-5.0, 5.0)]*self.dim)
                    if np.isnan(res.fun) or np.isinf(res.fun):
                        raise ValueError("Invalid function value")
                    if res.fun < 0 or res.fun > 1:
                        raise ValueError("Function value must be between 0 and 1")
                    res.fun_last = res.fun
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

class HEBBOWithCMAES(HEBBO):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.cmaes_scale = 1.0
        self.cmaes_offset = 0.0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            res = minimize(lambda x: -func(x), self.search_space, method="SLSQP", bounds=[(-5.0, 5.0)]*self.dim)
            if np.isnan(res.fun) or np.isinf(res.fun):
                raise ValueError("Invalid function value")
            if res.fun < 0 or res.fun > 1:
                raise ValueError("Function value must be between 0 and 1")
            if np.abs(res.fun - res.fun_last) < 1e-6:
                cmaes_scale = 1.0
                cmaes_offset = 0.0
            else:
                cmaes_scale = np.exp(-((res.fun - res.fun_last) / 1e-6) ** 2 / 2)
                cmaes_offset = np.random.normal(size=self.dim)
            res.fun_last = res.fun
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

# Example usage:
func = lambda x: np.sin(x)
hebbbo = HEBBO(100, 10)
hebbbo_value = hebbbo(func)
print(hebbbo_value)

hebbbo_with_adaptive_line_search = HEBBOWithAdaptiveLineSearch(100, 10)
hebbbo_with_adaptive_line_search_value = hebbbo_with_adaptive_line_search(func)
print(hebbbo_with_adaptive_line_search_value)

hebbbo_with_cmaes = HEBBOWithCMAES(100, 10)
hebbbo_with_cmaes_value = hebbbo_with_cmaes(func)
print(hebbbo_with_cmaes_value)