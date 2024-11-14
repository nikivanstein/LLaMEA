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
            
            for _ in range(self.budget - self.swarm_size):
                p_best = population[np.argmin(fitness)]
                r1, r2 = np.random.uniform(0, 1, (2, 1))
                velocity = self.inertia_weight * velocity + r1 * self.mutation_factor * (p_best - population.T) + r2 * (best_solution - population.T)
                population += velocity.T
                crossover_mask = np.random.uniform(0, 1, self.swarm_size) < self.crossover_prob
                candidate_idx = np.random.choice(range(self.swarm_size), (3, self.swarm_size), replace=True)
                candidate = population[candidate_idx]
                trial_vector = population + self.mutation_factor * (candidate[0] - candidate[1])
                np.place(trial_vector, candidate[2] < 0.5, candidate[2])
                trial_fitness = np.array([func(ind) for ind in trial_vector])
                better_idx = trial_fitness < fitness
                population[better_idx] = trial_vector[better_idx]
                fitness[better_idx] = trial_fitness[better_idx]
                best_idx = np.argmin(fitness)
                best_solution = population[best_idx]
            return best_solution
        return pso_de(func)