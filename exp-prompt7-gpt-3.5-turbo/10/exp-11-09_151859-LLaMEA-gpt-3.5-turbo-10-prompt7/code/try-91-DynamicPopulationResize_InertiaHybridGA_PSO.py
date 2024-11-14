import numpy as np

class DynamicPopulationResize_InertiaHybridGA_PSO:
    def __init__(self, budget, dim, population_size=50, mutation_rate=0.1, inertia_weight=0.5, cognitive_weight=1.5, social_weight=1.5):
        self.budget = budget
        self.dim = dim
        self.init_population_size = population_size
        self.mutation_rate = mutation_rate
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight

    def __call__(self, func):
        population_size = self.init_population_size
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        global_best = population[best_idx]
        velocity = np.zeros((population_size, self.dim))  # Initialize velocity

        for t in range(1, self.budget+1):
            # Update velocity based on PSO
            dynamic_inertia = self.inertia_weight * (1 - t/self.budget)  # Dynamic inertia weight
            velocity = dynamic_inertia * velocity + self.cognitive_weight * np.random.rand() * (best_individual - population) + self.social_weight * np.random.rand() * (global_best - population)
            population += velocity
            
            # Dynamically adjust mutation rate based on individual fitness
            mutation_rate_adjustment = 0.1 * np.exp(-np.mean(fitness) / np.max(fitness))  # Mutation rate based on mean fitness
            mutation_mask = np.random.rand(population_size, self.dim) < mutation_rate_adjustment
            population = population + np.random.uniform(-1.0, 1.0, (population_size, self.dim)) * mutation_mask
            
            fitness = np.array([func(ind) for ind in population])
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]
            global_best = population[np.argmin(fitness)]
            
            # Dynamic population resizing
            if t % (self.budget // 10) == 0:
                population_size = int(np.clip(population_size * 1.1, self.init_population_size, 1e6))
                population = np.vstack((population, np.random.uniform(-5.0, 5.0, (population_size - len(population), self.dim)))
                velocity = np.vstack((velocity, np.zeros((population_size - len(velocity), self.dim)))
        
        return global_best