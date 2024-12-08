class EnhancedFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.alpha = 0.2  # Initial alpha value
        self.alpha_min = 0.01
        self.alpha_max = 0.5  # Maximum allowed alpha value
        self.beta0 = 1.0
        self.gamma = 0.1
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population = self.initialize_population()
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if func(population[j]) < func(population[i]):
                        delta_alpha = (self.alpha - self.alpha_min) / (self.alpha_max - self.alpha_min)  # Dynamic adjustment of alpha
                        population[i] += delta_alpha * np.exp(-self.gamma * np.linalg.norm(population[j] - population[i])) * self.levy_flight() * self.chaotic_map(np.random.random())
                        evaluations += 1
                        if evaluations >= self.budget:
                            break

        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution