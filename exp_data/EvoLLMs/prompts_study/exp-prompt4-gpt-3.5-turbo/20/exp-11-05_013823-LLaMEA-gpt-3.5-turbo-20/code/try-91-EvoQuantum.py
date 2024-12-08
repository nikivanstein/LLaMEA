class EvoQuantum:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.mutation_rate = 0.1  # Initialize mutation rate

    def __call__(self, func):
        for _ in range(self.budget):
            fitness_values = np.array([func(individual) for individual in self.population])
            sorted_indices = np.argsort(fitness_values)
            elite = self.population[sorted_indices[:10]]  
            new_population = np.tile(elite, (10, 1)) 
            
            theta = np.random.uniform(0, 2*np.pi, (self.budget, self.dim))
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)])
            new_population = np.tensordot(new_population, rotation_matrix, axes=([1], [2]))
            
            # Update mutation rate adaptively based on diversity
            diversity = np.std(new_population)
            self.mutation_rate = 1.0 / (1.0 + np.exp(-diversity))  # Sigmoid-based adaptive mutation rate
            mutation_mask = np.random.choice([0, 1], size=(self.budget, self.dim), p=[1 - self.mutation_rate, self.mutation_rate])
            new_population += mutation_mask * np.random.normal(0, 1, (self.budget, self.dim))
            
            self.population = new_population
        best_solution = elite[0]  
        return func(best_solution)