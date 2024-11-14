import numpy as np

class EnhancedOptimizedPSODE:
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
                for j in range(self.swarm_size):
                    velocities[j] += self.inertia_weight * (r_value[j, 0] * self.mutation_factor * (p_best - population[j]) + r_value[j, 1] * (best_solution - population[j]))
                    population[j] += velocities[j]
                    if np.random.rand() < self.crossover_prob:
                        candidate_idxs = np.random.choice(range(self.swarm_size), 3, replace=False)
                        candidate = population[candidate_idxs]
                        trial_vector = population[j] + self.mutation_factor * (candidate[0] - candidate[1])
                        np.place(trial_vector, candidate[2] < 0.5, candidate[2])
                        trial_fitness = func(trial_vector)
                        if trial_fitness < fitness[j]:
                            population[j], fitness[j] = trial_vector, trial_fitness
                best_idx = np.argmin(fitness)
                best_solution = population[best_idx]
            return best_solution
        return pso_de(func)