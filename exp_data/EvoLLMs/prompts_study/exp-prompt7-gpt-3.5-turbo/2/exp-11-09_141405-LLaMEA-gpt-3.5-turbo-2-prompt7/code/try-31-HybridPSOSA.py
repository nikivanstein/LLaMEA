import numpy as np

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
            T_min, T_max = 0.01, 1.0
            decay_rate = 0.99
            for _ in range(self.budget):
                for i in range(self.dim):
                    new_position = mutate(swarm_position[i], T)
                    new_fitness = objective_function(new_position)
                    if new_fitness < objective_function(swarm_position[i]):
                        swarm_position[i] = new_position
                        if new_fitness < swarm_best_fitness:
                            swarm_best_position = new_position
                            swarm_best_fitness = new_fitness
                    T = max(T_min, min(T * decay_rate, T_max))
            return swarm_best_position

        return pso_sa_optimization()