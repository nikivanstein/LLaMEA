import numpy as np

class AcceleratedDynamicInertiaHybridGA_PSO:
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
        prev_best_individual = np.copy(best_individual)

        for t in range(1, self.budget+1):
            dynamic_inertia = self.inertia_weight * (1 - t/self.budget)  # Dynamic inertia weight
            velocity = dynamic_inertia * velocity + self.cognitive_weight * np.random.rand() * (best_individual - population) + self.social_weight * np.random.rand() * (global_best - population)
            
            # Additional velocity update
            additional_velocity = 0.2 * (best_individual - prev_best_individual)  # Encourage exploration towards promising regions
            velocity += additional_velocity
            
            population += velocity
            mutation_rate_adjustment = 0.1 * np.exp(-np.mean(fitness) / np.max(fitness))  
            mutation_mask = np.random.rand(self.population_size, self.dim) < mutation_rate_adjustment
            population = population + np.random.uniform(-1.0, 1.0, (self.population_size, self.dim)) * mutation_mask
            
            fitness = np.array([func(ind) for ind in population])
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]
            global_best = population[np.argmin(fitness)]
            prev_best_individual = np.copy(best_individual)
        
        return global_best