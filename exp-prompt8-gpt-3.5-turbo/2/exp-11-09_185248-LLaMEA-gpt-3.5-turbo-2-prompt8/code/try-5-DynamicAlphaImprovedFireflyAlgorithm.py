class DynamicAlphaImprovedFireflyAlgorithm(ImprovedFireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.alpha_min = 0.05
        self.alpha_max = 0.2

    def __call__(self, func):
        population = self.initialize_population()
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                for j in range(self.population_size):
                    alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * (evaluations / self.budget)  # Dynamic alpha scaling
                    if func(population[j]) < func(population[i]):
                        population[i] += alpha * np.exp(-self.gamma * np.linalg.norm(population[j] - population[i])) * self.levy_flight()
                        evaluations += 1
                        if evaluations >= self.budget:
                            break

        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution