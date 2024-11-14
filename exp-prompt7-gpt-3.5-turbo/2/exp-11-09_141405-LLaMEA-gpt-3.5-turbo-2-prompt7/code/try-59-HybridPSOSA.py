import numpy as np
from joblib import Parallel, delayed

class HybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def objective_function(x):
            return func(x)

        def initialize_population():
            return np.random.uniform(-5.0, 5.0, (self.dim,))

        def mutate(x, T):
            return x + T * np.random.normal(0, 1, x.shape)

        def pso_sa_optimization():
            swarm_position = initialize_population()
            swarm_best_position = swarm_position.copy()
            swarm_best_fitness = objective_function(swarm_best_position)
            T = 1.0
            for _ in range(self.budget):
                results = Parallel(n_jobs=-1)(delayed(objective_function)(swarm_position[i]) for i in range(self.dim))
                for i in range(self.dim):
                    new_position = mutate(swarm_position[i], T)
                    new_fitness = results[i]
                    if new_fitness < objective_function(swarm_position[i]):
                        swarm_position[i] = new_position
                        if new_fitness < swarm_best_fitness:
                            swarm_best_position = new_position
                            swarm_best_fitness = new_fitness
                T *= 0.99  # Simulated Annealing cooling schedule
            return swarm_best_position

        return pso_sa_optimization()