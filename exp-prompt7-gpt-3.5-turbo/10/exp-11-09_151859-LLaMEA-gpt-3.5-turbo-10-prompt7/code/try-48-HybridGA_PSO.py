import numpy as np

class HybridGA_PSO:
    def __init__(self, budget, dim, population_size=50, mutation_rate=0.1, initial_inertia_weight=0.5, cognitive_weight=1.5, social_weight=1.5, inertia_decay=0.95):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.inertia_weight = initial_inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.inertia_decay = inertia_decay

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        for _ in range(self.budget):
            # Update velocity based on PSO with dynamic inertia weight
            velocity = self.inertia_weight * velocity + self.cognitive_weight * np.random.rand() * (best_individual - population) + self.social_weight * np.random.rand() * (global_best - population)
            population += velocity
            
            # Mutate based on GA
            mutation_mask = np.random.rand(self.population_size, self.dim) < self.mutation_rate
            population = population + np.random.uniform(-1.0, 1.0, (self.population_size, self.dim)) * mutation_mask
            
            fitness = np.array([func(ind) for ind in population])
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]
            global_best = population[np.argmin(fitness)]
            self.inertia_weight *= self.inertia_decay
        
        return global_best