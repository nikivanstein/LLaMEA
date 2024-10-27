import numpy as np

class PSOADEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.inertia_weight = 0.5
        self.c1 = 1.5
        self.c2 = 2.0
        self.mutation_rate = 0.5

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)

        for _ in range(self.budget // self.population_size):
            swarm = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.population_size)]
            velocities = [np.zeros(self.dim) for _ in range(self.population_size)]

            for idx in range(self.population_size):
                fitness = func(swarm[idx])
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = swarm[idx]

                if np.random.rand() < self.mutation_rate:
                    ind1, ind2, ind3 = np.random.choice(range(self.population_size), 3, replace=False)
                    mutant = swarm[idx] + self.c1 * np.random.rand() * (best_solution - swarm[idx]) + self.c2 * np.random.rand() * (swarm[ind1] - swarm[ind2])
                    trial = np.where(np.random.uniform(0, 1, self.dim) < 0.5, mutant, swarm[idx])
                    trial_fitness = func(trial)
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                velocities[idx] = self.inertia_weight * velocities[idx] + self.c1 * np.random.rand() * (best_solution - swarm[idx]) + self.c2 * np.random.rand() * (swarm[idx] - swarm[ind1])
                swarm[idx] = np.clip(swarm[idx] + velocities[idx], -5.0, 5.0)

        return best_solution