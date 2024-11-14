import numpy as np

class HybridGA_PSO_Elitism:
    def __init__(self, budget, dim, population_size=50, mutation_rate=0.1, inertia_weight=0.5, cognitive_weight=1.5, social_weight=1.5, elitism_ratio=0.1):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.elitism_ratio = elitism_ratio

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
            
            # Dynamically adjust mutation rate
            mutation_rate_adjustment = 0.1 * np.exp(-_ / self.budget)  # Exponential decay of mutation rate
            mutation_mask = np.random.rand(self.population_size, self.dim) < mutation_rate_adjustment
            population = population + np.random.uniform(-1.0, 1.0, (self.population_size, self.dim)) * mutation_mask
            
            fitness = np.array([func(ind) for ind in population])
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]
            global_best = population[np.argmin(fitness)]
            
            # Elitism: Retain top individuals
            num_elites = int(self.elitism_ratio * self.population_size)
            elite_idx = np.argsort(fitness)[:num_elites]
            population[elite_idx] = population[elite_idx]

        return global_best