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
            inertia_weight = 0.9
            for _ in range(self.budget):
                for i in range(self.dim):
                    new_position = mutate(swarm_position[i], inertia_weight)
                    new_fitness = objective_function(new_position)
                    if new_fitness < objective_function(swarm_position[i]):
                        swarm_position[i] = new_position
                        if new_fitness < swarm_best_fitness:
                            swarm_best_position = new_position
                            swarm_best_fitness = new_fitness
                inertia_weight *= 0.98  # Dynamic inertia weight update
            return swarm_best_position

        return pso_sa_optimization()