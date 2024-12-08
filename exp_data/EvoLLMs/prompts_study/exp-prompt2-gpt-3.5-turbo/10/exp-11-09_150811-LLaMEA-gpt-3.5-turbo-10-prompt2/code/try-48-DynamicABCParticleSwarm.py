import numpy as np

class DynamicABCParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        swarm_size = 10
        inertia_weight = 0.5
        cognitive_weight = 1.5
        social_weight = 1.5
        best_global_solution = np.random.uniform(-5.0, 5.0, self.dim)
        
        for _ in range(self.budget):
            fitness = [func(x) for x in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]

            for i in range(self.budget):
                if i != best_idx:
                    cognitive_velocity = cognitive_weight * np.random.uniform(0, 1, self.dim) * (best_solution - self.population[i])
                    social_velocity = social_weight * np.random.uniform(0, 1, self.dim) * (best_global_solution - self.population[i])
                    velocity = inertia_weight * cognitive_velocity + social_velocity
                    trial_solution = self.population[i] + velocity
                    if func(trial_solution) < fitness[i]:
                        self.population[i] = trial_solution
                        
            best_global_idx = np.argmin([func(x) for x in self.population])
            best_global_solution = self.population[best_global_idx]
        
        return best_global_solution