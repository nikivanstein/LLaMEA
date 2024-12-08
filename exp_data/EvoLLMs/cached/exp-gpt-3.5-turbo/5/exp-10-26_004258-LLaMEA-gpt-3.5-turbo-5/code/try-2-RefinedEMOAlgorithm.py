import numpy as np
from scipy.optimize import differential_evolution
from cma import CMAEvolutionStrategy

class RefinedEMOAlgorithm:
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
            es = CMAEvolutionStrategy(x0, 0.5, {'bounds': [-5, 5], 'seed': 42})
            for _ in range(self.budget):
                solutions = es.ask()
                values = [func(sol) for sol in solutions]
                es.tell(solutions, values)
            return es.result.xbest, es.result.fbest

        def custom_mutation(x):
            mutated = x + np.random.normal(0, 0.1, size=x.shape)
            return np.clip(mutated, -5, 5)

        def optimize_custom_mutation(x0):
            best_x = x0
            best_f = func(x0)
            for _ in range(self.budget):
                new_x = custom_mutation(best_x)
                new_f = func(new_x)
                if new_f < best_f:
                    best_x, best_f = new_x, new_f
            return best_x, best_f

        x0 = np.random.uniform(-5, 5, self.dim)
        
        de_result = optimize_de(x0)
        cma_es_result = optimize_cma_es(x0)
        custom_mutation_result = optimize_custom_mutation(x0)
        
        return min(de_result, cma_es_result, custom_mutation_result, key=lambda x: x[1])

RefinedEMOAlgorithm(1000, 10)(func)  # Example of usage with function 'func'