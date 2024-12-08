import numpy as np
from scipy.optimize import differential_evolution

class HEACOMBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        if func.__name__ == 'f1':
            return differential_evolution(func, self.bounds)
        elif func.__name__ == 'f2':
            return differential_evolution(func, self.bounds, x0=self.x0)
        elif func.__name__ == 'f3':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5)
        elif func.__name__ == 'f4':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0)
        elif func.__name__ == 'f5':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx')
        elif func.__name__ == 'f6':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5)
        elif func.__name__ == 'f7':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5)
        elif func.__name__ == 'f8':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5)
        elif func.__name__ == 'f9':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1')
        elif func.__name__ == 'f10':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform')
        elif func.__name__ == 'f11':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='log-uniform')
        elif func.__name__ == 'f12':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='log-uniform', scaling='uniform')
        elif func.__name__ == 'f13':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform')
        elif func.__name__ == 'f14':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform')
        elif func.__name__ == 'f15':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f16':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f17':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f18':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f19':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f20':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f21':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f22':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f23':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f24':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')

    def refine_strategy(self, func):
        if self.budget > 0:
            best_individual = np.array(self.f(func))
            self.budget -= 1
            new_individual = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
            while np.any(new_individual == best_individual):
                new_individual = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
            self.x0 = new_individual
            return new_individual
        else:
            return np.nan, np.nan

    def evaluate_fitness(self, individual):
        func_name = individual.__class__.__name__
        if func_name == 'f1':
            return self.f(func_name)
        elif func_name == 'f2':
            return self.f(func_name, x0=individual)
        elif func_name == 'f3':
            return self.f(func_name, tol=1e-5)
        elif func_name == 'f4':
            return self.f(func_name, x0_init=individual)
        elif func_name == 'f5':
            return self.f(func_name, strategy='sbx')
        elif func_name == 'f6':
            return self.f(func_name, cxpb=0.5)
        elif func_name == 'f7':
            return self.f(func_name, cxpb=0.5, mu=0.5)
        elif func_name == 'f8':
            return self.f(func_name, cxpb=0.5, mu=0.5, w=0.5)
        elif func_name == 'f9':
            return self.f(func_name, cxpb=0.5, mu=0.5, w=0.5, strategy='rand1')
        elif func_name == 'f10':
            return self.f(func_name, cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform')
        elif func_name == 'f11':
            return self.f(func_name, cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='log-uniform')
        elif func_name == 'f12':
            return self.f(func_name, cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='log-uniform', scaling='uniform')
        elif func_name == 'f13':
            return self.f(func_name, cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform')
        elif func_name == 'f14':
            return self.f(func_name, cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform')
        elif func_name == 'f15':
            return self.f(func_name, cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        elif func_name == 'f16':
            return self.f(func_name, cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        elif func_name == 'f17':
            return self.f(func_name, cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func_name == 'f18':
            return self.f(func_name, cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func_name == 'f19':
            return self.f(func_name, cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func_name == 'f20':
            return self.f(func_name, cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func_name == 'f21':
            return self.f(func_name, cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func_name == 'f22':
            return self.f(func_name, cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func_name == 'f23':
            return self.f(func_name, cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func_name == 'f24':
            return self.f(func_name, cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')

    def optimize(self, func):
        if self.budget > 0:
            best_individual = np.array(self.f(func))
            self.budget -= 1
            new_individual = self.refine_strategy(func)
            return new_individual
        else:
            return np.nan, np.nan

# Usage
if __name__ == "__main__":
    bbo_functions = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24']
    for func in bbo_functions:
        bbo = HEACOMBBO(100, 10)
        best_individual, best_fitness = bbo.optimize(func)
        print(f"Function: {func}, Best Individual: {best_individual}, Best Fitness: {best_fitness}")