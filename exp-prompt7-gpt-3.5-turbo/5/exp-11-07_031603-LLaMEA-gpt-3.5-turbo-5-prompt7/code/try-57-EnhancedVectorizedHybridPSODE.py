import numpy as np

class EnhancedVectorizedHybridPSODE:
    def __init__(self, budget, dim, swarm_size=20, mutation_factor=0.5, crossover_prob=0.9, inertia_weight=0.5):
        self.budget, self.dim, self.swarm_size, self.mutation_factor, self.crossover_prob, self.inertia_weight = budget, dim, swarm_size, mutation_factor, crossover_prob, inertia_weight

    def __call__(self, func):
        def pso_de(func):
            population = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
            fitness = np.array([func(ind) for ind in population])
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]

            for _ in range(self.budget - self.swarm_size):
                p_best = population[np.argmin(fitness)]
                r_values = np.random.rand(self.swarm_size, 2)
                velocities = self.inertia_weight * np.zeros((self.swarm_size, self.dim))
                velocities += self.inertia_weight * (r_values[:, 0] * self.mutation_factor * (p_best - population) + r_values[:, 1] * (best_solution - population))
                population += velocities
                crossover_mask = np.random.rand(self.swarm_size) < self.crossover_prob
                candidate_idxs = np.random.choice(range(self.swarm_size), (self.swarm_size, 3), replace=True)
                candidate = population[candidate_idxs]
                trial_vectors = population + self.mutation_factor * (candidate[:, :, 0] - candidate[:, :, 1])
                np.putmask(trial_vectors, candidate[:, :, 2] < 0.5, candidate[:, :, 2])
                trial_fitness = np.array([func(tv) for tv in trial_vectors])
                improve_mask = trial_fitness < fitness
                population[improve_mask], fitness[improve_mask] = trial_vectors[improve_mask], trial_fitness[improve_mask]
                best_idx = np.argmin(fitness)
                best_solution = population[best_idx]
            return best_solution
        return pso_de(func)