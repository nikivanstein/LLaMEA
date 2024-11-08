import numpy as np

class EnhancedVectorizedHybridPSODE:
    def __init__(self, budget, dim, swarm_size=20, mutation_factor=0.5, crossover_prob=0.9, inertia_weight=0.5):
        self.budget, self.dim, self.swarm_size, self.mutation_factor, self.crossover_prob, self.inertia_weight = budget, dim, swarm_size, mutation_factor, crossover_prob, inertia_weight
        self.r_values = np.random.rand(budget, swarm_size, 2)

    def __call__(self, func):
        def pso_de(func):
            population = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
            fitness = np.array([func(ind) for ind in population])
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]

            for i in range(self.budget - self.swarm_size):
                p_best = population[np.argmin(fitness)]
                r_value = self.r_values[i % self.budget]
                velocities = self.inertia_weight * np.zeros((self.swarm_size, self.dim))
                velocities += self.inertia_weight * (r_value[:, 0][:, None] * self.mutation_factor * (p_best - population) + r_value[:, 1][:, None] * (best_solution - population))
                population += velocities
                crossover_mask = np.random.rand(self.swarm_size) < self.crossover_prob
                candidate_idxs = np.random.choice(range(self.swarm_size), (np.sum(crossover_mask), 3), replace=True)
                candidate = population[candidate_idxs]
                trial_vector = population[crossover_mask] + self.mutation_factor * (candidate[:, 0] - candidate[:, 1])
                np.place(trial_vector, candidate[:, 2] < 0.5, candidate[:, 2])
                trial_fitness = np.array([func(vec) for vec in trial_vector])
                update_mask = trial_fitness < fitness[crossover_mask]
                population[crossover_mask][update_mask] = trial_vector[update_mask]
                fitness[crossover_mask][update_mask] = trial_fitness[update_mask]
                best_idx = np.argmin(fitness)
                best_solution = population[best_idx]
            return best_solution
        return pso_de(func)