import numpy as np
from scipy.optimize import differential_evolution
import cma

class EMOAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def objective(x):
            return func(x)

        def optimize_de(x0):
            bounds = [(-5, 5)] * self.dim
            result = differential_evolution(objective, bounds, maxiter=self.budget, seed=42, popsize=10, tol=0.01)
            return result.x, result.fun

        def optimize_cma_es(x0):
            es = cma.CMAEvolutionStrategy(x0, 0.5, {'bounds': [-5, 5], 'seed': 42})
            for _ in range(self.budget):
                solutions = es.ask()
                values = [func(sol) for sol in solutions]
                es.tell(solutions, values)
            return es.result.xbest, es.result.fbest

        x0 = np.random.uniform(-5, 5, self.dim)
        
        de_result = optimize_de(x0)
        cma_es_result = optimize_cma_es(x0)
        
        return de_result if de_result[1] < cma_es_result[1] else cma_es_result