import numpy as np

class HybridAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.scale_factor = 0.5
        self.crossover_prob = 0.9
        self.inertia_weight = 0.7
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.velocity = np.zeros((budget, dim))
        self.best_position = self.population.copy()

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = self.population[a] + self.scale_factor * (self.population[b] - self.population[c])
                crossover_mask = np.random.rand(self.dim) < self.crossover_prob
                trial = np.where(crossover_mask, mutant, self.population[i])
                fitness_trial = func(trial)
                
                # Particle Swarm Optimization update
                self.velocity[i] = self.inertia_weight*self.velocity[i] + \
                                   self.cognitive_weight*np.random.rand()*(self.best_position[i] - self.population[i]) + \
                                   self.social_weight*np.random.rand()*(self.best_position[np.argmin([func(x) for x in self.population])] - self.population[i])
                self.population[i] = self.population[i] + self.velocity[i]
                
                if fitness_trial < func(self.population[i]):
                    self.population[i] = trial
                    if fitness_trial < func(self.best_position[i]):
                        self.best_position[i] = trial
                        
        final_fitness = [func(x) for x in self.population]
        best_idx = np.argmin(final_fitness)
        best_solution = self.population[best_idx]

        return best_solution