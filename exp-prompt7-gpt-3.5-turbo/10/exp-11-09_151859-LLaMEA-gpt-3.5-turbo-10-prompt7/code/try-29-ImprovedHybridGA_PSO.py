import numpy as np

class ImprovedHybridGA_PSO:
    def __init__(self, budget, dim, population_size=50, mutation_rate=0.1, inertia_weight=0.5, cognitive_weight=1.5, social_weight=1.5):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        velocity = np.zeros((self.population_size, self.dim))  # Initialize velocity
        
        for _ in range(self.budget):
            # Update velocity based on PSO with adaptive inertia weight
            adaptive_inertia_weight = 0.5 + 0.5 * np.cos((np.pi * _) / self.budget)
            velocity = adaptive_inertia_weight * velocity + self.cognitive_weight * np.random.rand() * (best_individual - population) + self.social_weight * np.random.rand() * (global_best - population)
            population += velocity
            
            # Mutate based on GA
            mutation_mask = np.random.rand(self.population_size, self.dim) < self.mutation_rate
            population = population + np.random.uniform(-1.0, 1.0, (self.population_size, self.dim)) * mutation_mask
            
            fitness = np.array([func(ind) for ind in population])
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]
            global_best = population[np.argmin(fitness)]
        
        return global_best