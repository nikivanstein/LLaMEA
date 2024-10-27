import numpy as np
import random

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
            return self.f1(func)
        elif func.__name__ == 'f2':
            return self.f2(func)
        elif func.__name__ == 'f3':
            return self.f3(func)
        elif func.__name__ == 'f4':
            return self.f4(func)
        elif func.__name__ == 'f5':
            return self.f5(func)
        elif func.__name__ == 'f6':
            return self.f6(func)
        elif func.__name__ == 'f7':
            return self.f7(func)
        elif func.__name__ == 'f8':
            return self.f8(func)
        elif func.__name__ == 'f9':
            return self.f9(func)
        elif func.__name__ == 'f10':
            return self.f10(func)
        elif func.__name__ == 'f11':
            return self.f11(func)
        elif func.__name__ == 'f12':
            return self.f12(func)
        elif func.__name__ == 'f13':
            return self.f13(func)
        elif func.__name__ == 'f14':
            return self.f14(func)
        elif func.__name__ == 'f15':
            return self.f15(func)
        elif func.__name__ == 'f16':
            return self.f16(func)
        elif func.__name__ == 'f17':
            return self.f17(func)
        elif func.__name__ == 'f18':
            return self.f18(func)
        elif func.__name__ == 'f19':
            return self.f19(func)
        elif func.__name__ == 'f20':
            return self.f20(func)
        elif func.__name__ == 'f21':
            return self.f21(func)
        elif func.__name__ == 'f22':
            return self.f22(func)
        elif func.__name__ == 'f23':
            return self.f23(func)
        elif func.__name__ == 'f24':
            return self.f24(func)

    def f1(self, func):
        # Differential Evolution
        res = differential_evolution(func, self.bounds)
        return res.x, res.fun

    def f2(self, func):
        # Differential Evolution with initial guess
        res = differential_evolution(func, self.bounds, x0=self.x0)
        return res.x, res.fun

    def f3(self, func):
        # Differential Evolution with initial guess and tolerance
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5)
        return res.x, res.fun

    def f4(self, func):
        # Differential Evolution with initial guess, tolerance, and scaling
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, scaling='uniform')
        return res.x, res.fun

    def f5(self, func):
        # Differential Evolution with initial guess, tolerance, scaling, and strategy
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, scaling='log-uniform', strategy='sbx')
        return res.x, res.fun

    def f6(self, func):
        # Differential Evolution with initial guess, tolerance, scaling, strategy, and crossover probability
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, scaling='log-uniform', strategy='sbx', cxpb=0.5)
        return res.x, res.fun

    def f7(self, func):
        # Differential Evolution with initial guess, tolerance, scaling, strategy, crossover probability, and mutation probability
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, scaling='log-uniform', strategy='sbx', cxpb=0.5, mu=0.5)
        return res.x, res.fun

    def f8(self, func):
        # Differential Evolution with initial guess, tolerance, scaling, strategy, crossover probability, mutation probability, and weight
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, scaling='log-uniform', strategy='sbx', cxpb=0.5, mu=0.5, w=0.5)
        return res.x, res.fun

    def f9(self, func):
        # Differential Evolution with initial guess, tolerance, scaling, strategy, crossover probability, mutation probability, weight, and scaling
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, scaling='log-uniform', strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1')
        return res.x, res.fun

    def f10(self, func):
        # Differential Evolution with initial guess, tolerance, scaling, strategy, crossover probability, mutation probability, weight, scaling, and scaling
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, scaling='log-uniform', strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform')
        return res.x, res.fun

    def f11(self, func):
        # Differential Evolution with initial guess, tolerance, scaling, strategy, crossover probability, mutation probability, weight, scaling, strategy, and scaling
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, scaling='log-uniform', strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='log-uniform')
        return res.x, res.fun

    def f12(self, func):
        # Differential Evolution with initial guess, tolerance, scaling, strategy, crossover probability, mutation probability, weight, scaling, strategy, scaling, and scaling
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, scaling='log-uniform', strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform')
        return res.x, res.fun

    def f13(self, func):
        # Differential Evolution with initial guess, tolerance, scaling, strategy, crossover probability, mutation probability, weight, scaling, strategy, scaling, and scaling, and scaling
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, scaling='log-uniform', strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform')
        return res.x, res.fun

    def f14(self, func):
        # Differential Evolution with initial guess, tolerance, scaling, strategy, crossover probability, mutation probability, weight, scaling, strategy, scaling, scaling, and scaling
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, scaling='log-uniform', strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform')
        return res.x, res.fun

    def f15(self, func):
        # Differential Evolution with initial guess, tolerance, scaling, strategy, crossover probability, mutation probability, weight, scaling, strategy, scaling, scaling, scaling, and scaling
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, scaling='log-uniform', strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        return res.x, res.fun

    def f16(self, func):
        # Differential Evolution with initial guess, tolerance, scaling, strategy, crossover probability, mutation probability, weight, scaling, strategy, scaling, scaling, scaling, scaling, and scaling
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, scaling='log-uniform', strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        return res.x, res.fun

    def f17(self, func):
        # Differential Evolution with initial guess, tolerance, scaling, strategy, crossover probability, mutation probability, weight, scaling, strategy, scaling, scaling, scaling, scaling, and scaling, and scaling
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, scaling='log-uniform', strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        return res.x, res.fun

    def f18(self, func):
        # Differential Evolution with initial guess, tolerance, scaling, strategy, crossover probability, mutation probability, weight, scaling, strategy, scaling, scaling, scaling, scaling, and scaling, and scaling, and scaling
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, scaling='log-uniform', strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        return res.x, res.fun

    def f19(self, func):
        # Differential Evolution with initial guess, tolerance, scaling, strategy, crossover probability, mutation probability, weight, scaling, strategy, scaling, scaling, scaling, scaling, and scaling, and scaling, and scaling, and scaling
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, scaling='log-uniform', strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        return res.x, res.fun

    def f20(self, func):
        # Differential Evolution with initial guess, tolerance, scaling, strategy, crossover probability, mutation probability, weight, scaling, strategy, scaling, scaling, scaling, scaling, and scaling, and scaling, and scaling, and scaling, and scaling
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, scaling='log-uniform', strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        return res.x, res.fun

    def f21(self, func):
        # Differential Evolution with initial guess, tolerance, scaling, strategy, crossover probability, mutation probability, weight, scaling, strategy, scaling, scaling, scaling, scaling, and scaling, and scaling, and scaling, and scaling, and scaling, and scaling
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, scaling='log-uniform', strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        return res.x, res.fun

    def f22(self, func):
        # Differential Evolution with initial guess, tolerance, scaling, strategy, crossover probability, mutation probability, weight, scaling, strategy, scaling, scaling, scaling, scaling, and scaling, and scaling, and scaling, and scaling, and scaling, and scaling, and scaling
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, scaling='log-uniform', strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        return res.x, res.fun

    def f23(self, func):
        # Differential Evolution with initial guess, tolerance, scaling, strategy, crossover probability, mutation probability, weight, scaling, strategy, scaling, scaling, scaling, scaling, and scaling, and scaling, and scaling, and scaling, and scaling, and scaling, and scaling, and scaling
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, scaling='log-uniform', strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        return res.x, res.fun

    def f24(self, func):
        # Differential Evolution with initial guess, tolerance, scaling, strategy, crossover probability, mutation probability, weight, scaling, strategy, scaling, scaling, scaling, scaling, and scaling, and scaling, and scaling, and scaling, and scaling, and scaling, and scaling, and scaling, and scaling
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, scaling='log-uniform', strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        return res.x, res.fun

    def refine_strategy(self, func):
        if self.budget > 0:
            res = self.f(func)
            if random.random() < 0.4:
                # Randomly select a strategy from the current population
                strategies = [self.f1, self.f2, self.f3, self.f4, self.f5, self.f6, self.f7, self.f8, self.f9, self.f10, self.f11, self.f12, self.f13, self.f14, self.f15, self.f16, self.f17, self.f18, self.f19, self.f20, self.f21, self.f22, self.f23, self.f24]
                strategy = random.choice(strategies)
                new_res = strategy(func)
                return new_res.x, new_res.fun
            else:
                return res.x, res.fun
        else:
            return np.nan, np.nan

# Usage
if __name__ == '__main__':
    # Initialize the algorithm with a budget of 100 and a dimension of 10
    algorithm = HEACOMBBO(budget=100, dim=10)
    # Optimize the function f1
    res = algorithm('f1')
    print("Optimal solution:", res.x)
    print("Optimal value:", res.fun)