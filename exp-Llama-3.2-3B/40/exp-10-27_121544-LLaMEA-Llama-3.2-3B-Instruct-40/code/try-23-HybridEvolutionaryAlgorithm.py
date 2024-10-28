import numpy as np
import random

class HybridEvolutionaryAlgorithm:
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
        # Differential Evolution with initial guess, tolerance, and initial guess
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0)
        return res.x, res.fun

    def f5(self, func):
        # Differential Evolution with initial guess, tolerance, initial guess, and strategy
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx')
        return res.x, res.fun

    def f6(self, func):
        # Differential Evolution with initial guess, tolerance, initial guess, strategy, and crossover probability
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5)
        return res.x, res.fun

    def f7(self, func):
        # Differential Evolution with initial guess, tolerance, initial guess, strategy, crossover probability, and mutation probability
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5)
        return res.x, res.fun

    def f8(self, func):
        # Differential Evolution with initial guess, tolerance, initial guess, strategy, crossover probability, mutation probability, and weighting
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5)
        return res.x, res.fun

    def f9(self, func):
        # Differential Evolution with initial guess, tolerance, initial guess, strategy, crossover probability, mutation probability, weighting, and scaling strategy
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1')
        return res.x, res.fun

    def f10(self, func):
        # Differential Evolution with initial guess, tolerance, initial guess, strategy, crossover probability, mutation probability, weighting, scaling strategy, and scaling method
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform')
        return res.x, res.fun

    def f11(self, func):
        # Differential Evolution with initial guess, tolerance, initial guess, strategy, crossover probability, mutation probability, weighting, scaling strategy, scaling method, and scaling method
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='log-uniform')
        return res.x, res.fun

    def f12(self, func):
        # Differential Evolution with initial guess, tolerance, initial guess, strategy, crossover probability, mutation probability, weighting, scaling strategy, scaling method, scaling method, and scaling method
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='log-uniform', scaling='uniform')
        return res.x, res.fun

    def f13(self, func):
        # Differential Evolution with initial guess, tolerance, initial guess, strategy, crossover probability, mutation probability, weighting, scaling strategy, scaling method, scaling method, scaling method, and scaling method
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform')
        return res.x, res.fun

    def f14(self, func):
        # Differential Evolution with initial guess, tolerance, initial guess, strategy, crossover probability, mutation probability, weighting, scaling strategy, scaling method, scaling method, scaling method, scaling method, and scaling method
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform')
        return res.x, res.fun

    def f15(self, func):
        # Differential Evolution with initial guess, tolerance, initial guess, strategy, crossover probability, mutation probability, weighting, scaling strategy, scaling method, scaling method, scaling method, scaling method, scaling method, and scaling method
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        return res.x, res.fun

    def f16(self, func):
        # Differential Evolution with initial guess, tolerance, initial guess, strategy, crossover probability, mutation probability, weighting, scaling strategy, scaling method, scaling method, scaling method, scaling method, scaling method, scaling method, and scaling method
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        return res.x, res.fun

    def f17(self, func):
        # Differential Evolution with initial guess, tolerance, initial guess, strategy, crossover probability, mutation probability, weighting, scaling strategy, scaling method, scaling method, scaling method, scaling method, scaling method, scaling method, scaling method, and scaling method
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        return res.x, res.fun

    def f18(self, func):
        # Differential Evolution with initial guess, tolerance, initial guess, strategy, crossover probability, mutation probability, weighting, scaling strategy, scaling method, scaling method, scaling method, scaling method, scaling method, scaling method, scaling method, scaling method, and scaling method
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        return res.x, res.fun

    def f19(self, func):
        # Differential Evolution with initial guess, tolerance, initial guess, strategy, crossover probability, mutation probability, weighting, scaling strategy, scaling method, scaling method, scaling method, scaling method, scaling method, scaling method, scaling method, scaling method, and scaling method
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        return res.x, res.fun

    def f20(self, func):
        # Differential Evolution with initial guess, tolerance, initial guess, strategy, crossover probability, mutation probability, weighting, scaling strategy, scaling method, scaling method, scaling method, scaling method, scaling method, scaling method, scaling method, scaling method, and scaling method
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        return res.x, res.fun

    def f21(self, func):
        # Differential Evolution with initial guess, tolerance, initial guess, strategy, crossover probability, mutation probability, weighting, scaling strategy, scaling method, scaling method, scaling method, scaling method, scaling method, scaling method, scaling method, scaling method, and scaling method
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        return res.x, res.fun

    def f22(self, func):
        # Differential Evolution with initial guess, tolerance, initial guess, strategy, crossover probability, mutation probability, weighting, scaling strategy, scaling method, scaling method, scaling method, scaling method, scaling method, scaling method, scaling method, scaling method, and scaling method
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        return res.x, res.fun

    def f23(self, func):
        # Differential Evolution with initial guess, tolerance, initial guess, strategy, crossover probability, mutation probability, weighting, scaling strategy, scaling method, scaling method, scaling method, scaling method, scaling method, scaling method, scaling method, scaling method, and scaling method
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        return res.x, res.fun

    def f24(self, func):
        # Differential Evolution with initial guess, tolerance, initial guess, strategy, crossover probability, mutation probability, weighting, scaling strategy, scaling method, scaling method, scaling method, scaling method, scaling method, scaling method, scaling method, scaling method, and scaling method
        res = differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        return res.x, res.fun

    def refine(self, func):
        # Refine the solution using probability-based refinement
        refined_x = []
        for individual in self.x0:
            if random.random() < 0.4:
                # Apply mutation
                individual = (individual + random.uniform(-0.1, 0.1),)
                refined_x.append(individual)
            else:
                refined_x.append(individual)
        self.x0 = refined_x
        return self.x0

# Example usage
if __name__ == "__main__":
    algorithm = HybridEvolutionaryAlgorithm(100, 10)
    # Replace 'f1' with the desired function from the BBOB test suite
    algorithm('f1')