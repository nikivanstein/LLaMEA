import numpy as np

class FireflyOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_fireflies = 10
        self.alpha = 0.2
        self.beta = 1.0

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.num_fireflies, self.dim))

        def evaluate_population(population):
            return np.array([func(solution) for solution in population])

        def attractiveness(distance):
            return self.alpha * np.exp(-self.beta * distance)

        def update_population(population, fitness):
            for i in range(self.num_fireflies):
                for j in range(self.num_fireflies):
                    if fitness[j] < fitness[i]:
                        distance = np.linalg.norm(population[i] - population[j])
                        attractiveness_value = attractiveness(distance)
                        population[i] += attractiveness_value * (population[j] - population[i])
                        population[i] = np.clip(population[i], self.lower_bound, self.upper_bound)
                        fitness[i] = func(population[i])

            return population, fitness

        population = initialize_population()
        fitness = evaluate_population(population)

        for _ in range(self.budget - self.budget // 10):
            population, fitness = update_population(population, fitness)

        best_idx = np.argmin(fitness)
        return population[best_idx]