import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lower_bound = -5.0
        upper_bound = 5.0
        population_size = 10
        swarm = np.random.uniform(low=lower_bound, high=upper_bound, size=(population_size, self.dim))
        best_swarm_fitness = np.inf
        best_swarm_position = swarm[0]
        for _ in range(self.budget):
            for i in range(population_size):
                mutant = swarm[np.random.choice(population_size, 3, replace=False)]
                trial_vector = swarm[i] + 0.8 * (mutant[0] - mutant[1]) + 0.5 * (mutant[2] - swarm[i])
                trial_vector = np.clip(trial_vector, lower_bound, upper_bound)
                trial_fitness = func(trial_vector)
                if trial_fitness < best_swarm_fitness:
                    best_swarm_fitness = trial_fitness
                    best_swarm_position = trial_vector
                    swarm[i] = trial_vector
        return best_swarm_position