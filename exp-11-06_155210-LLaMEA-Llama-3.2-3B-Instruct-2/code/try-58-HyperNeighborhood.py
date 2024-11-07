class HyperNeighborhood:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []
        self.fitness_values = []
        self.best_solution = None

    def __call__(self, func):
        # Initialize the population with random solutions
        for _ in range(self.budget):
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            self.population.append(solution)
            self.fitness_values.append(func(solution))

        # Initialize the best solution and its fitness value
        self.best_solution = self.population[0]
        self.best_fitness = self.fitness_values[0]

        # Iterate until the budget is exhausted
        for _ in range(self.budget):
            # Select the best solution and its fitness value
            best_idx = self.fitness_values.index(self.best_fitness)
            for i in range(self.budget):
                if self.fitness_values[i] > self.best_fitness:
                    best_idx = i
            best_solution = self.population[best_idx]
            best_fitness = self.fitness_values[best_idx]

            # Create a new solution by perturbing the best solution
            new_solution = best_solution + np.random.uniform(-0.1, 0.1, self.dim)

            # Evaluate the new solution
            new_fitness = func(new_solution)

            # Replace the best solution if the new solution is better
            if new_fitness < self.best_fitness:
                self.best_solution = new_solution
                self.best_fitness = new_fitness

            # Replace the least fit solution with the new solution
            self.fitness_values[self.fitness_values.index(min(self.fitness_values))] = new_fitness
            self.population[self.fitness_values.index(min(self.fitness_values))] = new_solution