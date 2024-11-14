import numpy as np

class EnhancedVectorizedHybridPSODE:
    def __init__(self, budget, dim, swarm_size=20, mutation_factor=0.5, crossover_prob=0.9, inertia_weight=0.5):
        self.budget, self.dim, self.swarm_size, self.mutation_factor, self.crossover_prob, self.inertia_weight = budget, dim, swarm_size, mutation_factor, crossover_prob, inertia_weight
        self.r_values = np.random.rand(self.budget, self.swarm_size, 2)

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
                for i in range(self.swarm_size):
                    velocities[i] += self.inertia_weight * (r_value[i, 0] * self.mutation_factor * (p_best - population[i]) + r_value[i, 1] * (best_solution - population[i]))
                    population[i] += velocities[i]
                    if np.random.rand() < self.crossover_prob:
                        candidate_idxs = np.random.choice(range(self.swarm_size), 3, replace=False)
                        candidate = population[candidate_idxs]
                        trial_vector = population[i] + self.mutation_factor * (candidate[0] - candidate[1])
                        np.copyto(trial_vector, np.where(candidate[2] < 0.5, candidate[2], trial_vector))
                        trial_fitness = func(trial_vector)
                        if trial_fitness < fitness[i]:
                            population[i], fitness[i] = trial_vector, trial_fitness
                best_idx = np.argmin(fitness)
                best_solution = population[best_idx]
            return best_solution
        return pso_de(func)