class HybridGA_PSO:
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
        best_fitness = np.min(fitness)
        
        for _ in range(self.budget):
            fitness = np.array([func(ind) for ind in population])
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]
            global_best = population[np.argmin(fitness)]
            
            # Update velocity based on PSO with dynamic learning rate
            learning_rate = 1 / (1 + best_fitness - np.min(fitness))
            velocity = self.inertia_weight * velocity + learning_rate * (self.cognitive_weight * np.random.rand() * (best_individual - population) + self.social_weight * np.random.rand() * (global_best - population))
            population += velocity
            
            # Mutate based on GA
            mutation_mask = np.random.rand(self.population_size, self.dim) < self.mutation_rate
            population = population + np.random.uniform(-1.0, 1.0, (self.population_size, self.dim)) * mutation_mask
        
        return global_best