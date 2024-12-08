import numpy as np
from scipy.optimize import differential_evolution

class HybridEvolutionaryAlgorithm:
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

        def optimize_pso(x0):
            swarm_size = 10
            max_iter = self.budget
            inertia_weight = 0.5
            cognitive_weight = 1.0
            social_weight = 2.0
            bounds = [(-5, 5)] * self.dim

            swarm = np.random.uniform(-5, 5, (swarm_size, self.dim))
            velocities = np.zeros((swarm_size, self.dim))

            best_position = swarm[0]
            best_value = func(swarm[0])

            for _ in range(max_iter):
                for i in range(swarm_size):
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    cognitive = cognitive_weight * r1 * (best_position - swarm[i])
                    social = social_weight * r2 * (best_position - swarm[i])
                    velocities[i] = inertia_weight * velocities[i] + cognitive + social
                    swarm[i] = np.clip(swarm[i] + velocities[i], -5, 5)

                    value = func(swarm[i])
                    if value < best_value:
                        best_value = value
                        best_position = swarm[i]

            return best_position, best_value

        x0 = np.random.uniform(-5, 5, self.dim)

        de_result = optimize_de(x0)
        pso_result = optimize_pso(x0)

        return de_result if de_result[1] < pso_result[1] else pso_result