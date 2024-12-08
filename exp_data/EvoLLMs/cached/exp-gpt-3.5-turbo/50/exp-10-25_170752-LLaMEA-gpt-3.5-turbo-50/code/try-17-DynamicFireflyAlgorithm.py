class DynamicFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.alpha = 0.5
        self.beta_min = 0.2
        self.gamma = 1.0
        self.population = np.random.uniform(-5.0, 5.0, (50, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')

    def move_fireflies(self, func):
        for i in range(len(self.population)):
            for j in range(len(self.population)):
                if func(self.population[j]) < func(self.population[i]):
                    beta = self.beta_min + (1 - self.beta_min) * np.random.random()
                    self.population[i] += self.alpha * (self.population[j] - self.population[i]) * self.attractiveness(self.population[i], self.population[j]) + self.gamma * (beta * np.random.uniform(-5.0, 5.0, self.dim))

    def adapt_population(self, func):
        diversity = np.std([func(ind) for ind in self.population])
        if diversity < 0.1:
            self.population = np.vstack([self.population, np.random.uniform(-5.0, 5.0, (10, self.dim))])
        elif diversity > 0.5 and len(self.population) > 10:
            self.population = self.population[:len(self.population)//2]

    def __call__(self, func):
        for _ in range(self.budget):
            self.move_fireflies(func)
            self.adapt_population(func)
            for i in range(len(self.population)):
                fitness = func(self.population[i])
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = np.copy(self.population[i])
        return self.best_solution