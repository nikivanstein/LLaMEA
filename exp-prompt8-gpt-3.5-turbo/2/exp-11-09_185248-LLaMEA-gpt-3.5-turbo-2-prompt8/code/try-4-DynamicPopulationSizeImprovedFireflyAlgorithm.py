class DynamicPopulationSizeImprovedFireflyAlgorithm(ImprovedFireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.initial_population_size = self.population_size

    def __call__(self, func):
        population = self.initialize_population()
        evaluations = 0
        iteration = 0

        while evaluations < self.budget:
            for i in range(len(population)):
                for j in range(len(population)):
                    if func(population[j]) < func(population[i]):
                        population[i] += self.alpha * np.exp(-self.gamma * np.linalg.norm(population[j] - population[i])) * self.levy_flight()
                        evaluations += 1
                        if evaluations >= self.budget:
                            break
                iteration += 1

                if iteration % 10 == 0:
                    population = np.concatenate((population, self.initialize_population()), axis=0)
                    if len(population) > self.initial_population_size:
                        fitness_scores = [func(individual) for individual in population]
                        sorted_indices = np.argsort(fitness_scores)
                        population = population[sorted_indices[:self.initial_population_size]]

        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution