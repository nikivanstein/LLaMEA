class FastFruitFlyOptimizationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.step_size = 1.0

    def __call__(self, func):
        population_size = 10
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness_values = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget - population_size):
            mean_individual = np.mean(population, axis=0)
            new_individual = mean_individual + np.random.uniform(-1, 1, self.dim) * self.step_size
            new_fitness = func(new_individual)
            
            if new_fitness < np.max(fitness_values):
                max_idx = np.argmax(fitness_values)
                population[max_idx] = new_individual
                fitness_values[max_idx] = new_fitness
                self.step_size *= 1.1  # Increase step size for better exploration
                if np.random.rand() < 0.2:  # Introduce dynamic population size adaptation
                    new_individual = np.random.uniform(self.lb, self.ub, self.dim)
                    new_fitness = func(new_individual)
                    min_idx = np.argmin(fitness_values)
                    population[min_idx] = new_individual
                    fitness_values[min_idx] = new_fitness
            else:
                self.step_size *= 0.9  # Decrease step size for better exploitation
        
        best_idx = np.argmin(fitness_values)
        best_solution = population[best_idx]
        best_fitness = fitness_values[best_idx]
        
        return best_solution, best_fitness