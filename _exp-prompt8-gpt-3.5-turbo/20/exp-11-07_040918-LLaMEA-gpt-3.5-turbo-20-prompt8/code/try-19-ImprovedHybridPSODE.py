import numpy as np
from scipy.optimize import differential_evolution

class ImprovedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def objective(x):
            return func(x)

        bounds = [(-5.0, 5.0) for _ in range(self.dim)]
        popsize = 10
        num_particles = 5

        def pso_de_optimizer():
            bounds_array = np.array(bounds)
            pso_result = differential_evolution(objective, bounds=bounds_array, maxiter=5, popsize=popsize)
            best_pos = pso_result.x

            def init_population_array(init):
                if init is None:
                    return None
                return np.asarray(init)

            de_solver = differential_evolution.DifferentialEvolutionSolver(objective, bounds, args=(), maxiter=None, popsize=None, strategy='best1bin', tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=None, callback=None, disp=False, polish=True, init=best_pos, atol=1e-4, updating='immediate', workers=1, constraints=())

            de_solver.init_population_array = init_population_array.__get__(de_solver)
            de_solver.init_population_array(best_pos)

            for _ in range(self.budget // num_particles - 1):
                pop = [best_pos + np.random.normal(0, 1, self.dim) for _ in range(num_particles)]
                for agent in pop:
                    de_result = de_solver.solve()
                    if de_result.fun < objective(best_pos):
                        best_pos = de_result.x

            return best_pos

        return pso_de_optimizer()