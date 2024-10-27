import numpy as np

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 20
        self.c1 = 1.5
        self.c2 = 1.5
        self.inertia_weight = 0.9
        self.temperature = 10.0

    def __call__(self, func):
        swarm = np.random.uniform(low=-5.0, high=5.0, size=(self.swarm_size, self.dim))
        swarm_velocity = np.zeros((self.swarm_size, self.dim))
        swarm_best = swarm.copy()
        swarm_best_fitness = np.array([func(individual) for individual in swarm_best])
        global_best_idx = np.argmin(swarm_best_fitness)
        global_best = swarm_best[global_best_idx]
        global_best_fitness = swarm_best_fitness[global_best_idx]

        for _ in range(self.budget):
            for i in range(self.swarm_size):
                r1 = np.random.uniform(0, self.c1)
                r2 = np.random.uniform(0, self.c2)
                swarm_velocity[i] = self.inertia_weight * swarm_velocity[i] + r1 * (swarm_best[i] - swarm[i]) + r2 * (global_best - swarm[i])
                swarm[i] = swarm[i] + swarm_velocity[i]
                swarm[i] = np.clip(swarm[i], -5.0, 5.0)
                candidate_fitness = func(swarm[i])

                if candidate_fitness < swarm_best_fitness[i]:
                    swarm_best[i] = swarm[i]
                    swarm_best_fitness[i] = candidate_fitness
                    if candidate_fitness < global_best_fitness:
                        global_best = swarm[i]
                        global_best_fitness = candidate_fitness

            self.temperature *= 0.95  # Simulated Annealing temperature reduction
        
        return global_best, global_best_fitness