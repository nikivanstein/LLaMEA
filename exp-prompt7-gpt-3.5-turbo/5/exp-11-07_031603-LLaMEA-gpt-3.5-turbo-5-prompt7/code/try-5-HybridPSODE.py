import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim, swarm_size=20, mutation_factor=0.5, crossover_prob=0.9, inertia_weight=0.5):
        self.budget, self.dim, self.swarm_size, self.mutation_factor, self.crossover_prob, self.inertia_weight = budget, dim, swarm_size, mutation_factor, crossover_prob, inertia_weight
        self.velocity = np.zeros((self.swarm_size, self.dim))  # Initialize velocity
        
    def __call__(self, func):
        def pso_de(func):
            population = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
            fitness = np.array([func(ind) for ind in population])
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]
            
            for _ in range(self.budget - self.swarm_size):
                p_best = population[np.argmin(fitness)]
                
                r_vals = np.random.uniform(0, 1, (self.swarm_size, 2))
                velocity = self.inertia_weight * self.velocity + r_vals[:, 0, None] * self.mutation_factor * (p_best - population) + r_vals[:, 1, None] * (best_solution - population)
                population += velocity
                
                mask = np.random.uniform(0, 1, (self.swarm_size, self.dim))
                candidate_idx = np.random.choice(range(self.swarm_size), (self.swarm_size, 3), replace=False)
                candidate = population[candidate_idx]
                trial_vector = population + self.mutation_factor * (candidate[:, 0] - candidate[:, 1])
                trial_vector[candidate[:, 2] < 0.5] = candidate[:, 2]
                
                improved_mask = func(trial_vector) < fitness
                population[improved_mask] = trial_vector[improved_mask]
                fitness[improved_mask] = func(trial_vector)[improved_mask]
                
                best_idx = np.argmin(fitness)
                best_solution = population[best_idx]
            
            return best_solution
        
        return pso_de(func)