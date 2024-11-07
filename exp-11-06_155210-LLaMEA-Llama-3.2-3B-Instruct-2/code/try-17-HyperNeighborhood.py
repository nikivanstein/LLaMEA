class HyperNeighborhood:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []
        self.fitness_values = []
        self.crossover_probability = 0.5  # New line: Initialize crossover probability

    def __call__(self, func):
        # Initialize the population with random solutions
        for _ in range(self.budget):
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            self.population.append(solution)
            self.fitness_values.append(func(solution))

        # Initialize the best solution and its fitness value
        best_solution = self.population[0]
        best_fitness = self.fitness_values[0]

        # Iterate until the budget is exhausted
        for _ in range(self.budget):
            # Select the best solution and its fitness value
            for i in range(self.budget):
                if self.fitness_values[i] > best_fitness:
                    best_solution = self.population[i]
                    best_fitness = self.fitness_values[i]

            # Create a new solution by perturbing the best solution
            new_solution = best_solution + np.random.uniform(-0.1, 0.1, self.dim)

            # Evaluate the new solution
            new_fitness = func(new_solution)

            # Replace the best solution if the new solution is better
            if new_fitness < best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness

            # Replace the least fit solution with the new solution
            self.fitness_values[self.fitness_values.index(min(self.fitness_values))] = new_fitness
            self.population[self.fitness_values.index(min(self.fitness_values))] = new_solution

            # Apply crossover with probability 0.5
            if np.random.rand() < self.crossover_probability:
                # Select two random parents
                parent1 = self.population[np.random.randint(0, self.budget)]
                parent2 = self.population[np.random.randint(0, self.budget)]

                # Create a new solution by crossover
                child = (parent1 + parent2) / 2

                # Evaluate the new solution
                child_fitness = func(child)

                # Replace the least fit solution with the new solution
                self.fitness_values[self.fitness_values.index(min(self.fitness_values))] = child_fitness
                self.population[self.fitness_values.index(min(self.fitness_values))] = child