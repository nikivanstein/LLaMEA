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
        
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        global_best = population[best_idx]
        velocity = np.zeros((self.population_size, self.dim))  # Initialize velocity
        
        mutation_rate_adjustment = 0.1
        for _ in range(self.budget):
            # Update velocity based on PSO
            velocity = self.inertia_weight * velocity + self.cognitive_weight * np.random.rand() * (best_individual - population) + self.social_weight * np.random.rand() * (global_best - population)
            population += velocity
            
            # New adaptive mutation strategy
            mutation_rate_adjustment = 0.1 * np.exp(-_ / self.budget)  # Exponential decay of mutation rate
            mutation_mask = np.random.rand(self.population_size, self.dim) < mutation_rate_adjustment
            population = population + np.random.uniform(-1.0, 1.0, (self.population_size, self.dim)) * mutation_mask
            
            fitness = np.array([func(ind) for ind in population])
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]
            global_best = population[np.argmin(fitness)]
            
            # Adaptive mutation rate update based on population diversity
            mean_individual = np.mean(population, axis=0)
            std_individual = np.std(population, axis=0)
            mutation_rate_adjustment = np.clip(0.1 + 0.1 * np.mean(std_individual), 0.1, 0.9)
            mutation_mask = np.random.rand(self.population_size, self.dim) < mutation_rate_adjustment
            population = population + np.random.uniform(-1.0, 1.0, (self.population_size, self.dim)) * mutation_mask
        
        return global_best