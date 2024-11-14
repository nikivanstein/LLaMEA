import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim, swarm_size=20, mutation_factor=0.5, crossover_prob=0.9, inertia_weight=0.5):
        self.budget, self.dim, self.swarm_size, self.mutation_factor, self.crossover_prob, self.inertia_weight = budget, dim, swarm_size, mutation_factor, crossover_prob, inertia_weight
        
    def __call__(self, func):
        def pso_de(func):
            population = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
            fitness = np.array([func(ind) for ind in population])
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]
            velocity = np.zeros((self.swarm_size, self.dim))

            for _ in range(self.budget - self.swarm_size):
                p_best = population[np.argmin(fitness)]
                r1, r2 = np.random.uniform(0, 1, (2, self.dim))
                velocity = self.inertia_weight * velocity + r1[:, np.newaxis] * self.mutation_factor * (p_best - population) + r2[:, np.newaxis] * (best_solution - population)
                population += velocity
                crossover_mask = np.random.uniform(0, 1, self.swarm_size) < self.crossover_prob
                candidate_indices = np.random.choice(self.swarm_size, (3, self.swarm_size), replace=False)
                candidates = population[candidate_indices]
                trial_vectors = population + self.mutation_factor * (candidates[0] - candidates[1])
                np.place(trial_vectors, candidates[2] < 0.5, candidates[2])
                trial_fitness = np.array([func(ind) for ind in trial_vectors])
                update_indices = trial_fitness < fitness
                population[update_indices] = trial_vectors[update_indices]
                fitness[update_indices] = trial_fitness[update_indices]
                best_idx = np.argmin(fitness)
                best_solution = population[best_idx]

            return best_solution
        return pso_de(func)