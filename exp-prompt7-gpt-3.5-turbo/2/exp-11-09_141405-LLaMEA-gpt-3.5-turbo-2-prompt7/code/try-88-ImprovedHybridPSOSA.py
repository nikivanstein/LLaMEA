import numpy as np

class ImprovedHybridPSOSA:
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
            swarm_best_fitness = np.inf
            best_position = None
            inertia_weight = 0.9
            neighborhood_size = 5
            neighborhood_best_fitness = np.inf
            for _ in range(self.budget):
                swarm_position = initialize_population()
                for _ in range(self.dim):
                    new_position = mutate(swarm_position, inertia_weight)
                    new_fitness = objective_function(new_position)
                    if new_fitness < swarm_best_fitness:
                        swarm_best_fitness = new_fitness
                        best_position = new_position
                    if new_fitness < neighborhood_best_fitness:
                        neighborhood_best_fitness = new_fitness
                inertia_weight *= 0.995  # Dynamic update of inertia weight
                neighborhood_best_fitness = np.inf  # Reset neighborhood best fitness
            return best_position

        return pso_sa_optimization()