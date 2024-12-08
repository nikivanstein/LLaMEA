import numpy as np

class HybridFireflyAlgorithm:
    def __init__(self, budget, dim, alpha=0.5, beta_min=0.2, beta_max=1.0, gamma=0.1):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.gamma = gamma

    def __call__(self, func):
        def evaluate_solution(solution):
            return func(solution)

        def initialize_population():
            return np.random.uniform(-5.0, 5.0, (self.dim))

        def attractiveness(brightness, distance):
            return self.beta_min + (self.beta_max - self.beta_min) * np.exp(-self.gamma * distance**2)

        def move_fireflies(population):
            new_population = np.copy(population)
            for i, firefly in enumerate(population):
                for j, other_firefly in enumerate(population):
                    if evaluate_solution(other_firefly) < evaluate_solution(firefly):
                        distance = np.linalg.norm(firefly - other_firefly)
                        attractiveness_factor = attractiveness(evaluate_solution(firefly), distance)
                        new_population[i] += self.alpha * (other_firefly - firefly) * attractiveness_factor
            return new_population

        best_solution = None
        best_fitness = np.inf

        population = initialize_population()
        for _ in range(self.budget):
            population = move_fireflies(population)
            for firefly in population:
                fitness = evaluate_solution(firefly)
                if fitness < best_fitness:
                    best_solution = firefly
                    best_fitness = fitness

        return best_solution