import numpy as np
from concurrent.futures import ThreadPoolExecutor

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

        def update_position(swarm_position, inertia_weight):
            new_position = mutate(swarm_position, inertia_weight)
            new_fitness = objective_function(new_position)
            return new_position, new_fitness

        def pso_sa_optimization():
            swarm_best_fitness = np.inf
            best_position = None
            inertia_weight = 0.9
            with ThreadPoolExecutor() as executor:
                for _ in range(self.budget):
                    swarm_position = initialize_population()
                    futures = [executor.submit(update_position, swarm_position, inertia_weight) for _ in range(self.dim)]
                    results = [future.result() for future in futures]
                    for new_position, new_fitness in results:
                        if new_fitness < swarm_best_fitness:
                            swarm_best_fitness = new_fitness
                            best_position = new_position
                    inertia_weight *= 0.995  # Dynamic update of inertia weight
            return best_position

        return pso_sa_optimization()