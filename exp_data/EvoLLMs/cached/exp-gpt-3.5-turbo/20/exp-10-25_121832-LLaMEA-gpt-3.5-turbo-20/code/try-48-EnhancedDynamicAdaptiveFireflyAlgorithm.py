import numpy as np

class EnhancedDynamicAdaptiveFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.alpha = 0.2
        self.beta_min = 0.2
        self.beta_max = 1.0

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        def attractiveness(distance):
            return self.beta_min + (self.beta_max - self.beta_min) * np.exp(-self.alpha * distance)

        population = initialize_population()
        evaluations = 0

        while evaluations < self.budget:
            for i, x in enumerate(population):
                for j, y in enumerate(population):
                    if func(y) < func(x):
                        distance = np.linalg.norm(x - y)
                        beta = attractiveness(distance)
                        population[i] += beta * (y - x) + np.random.uniform(-1, 1, self.dim)
                        
                        # Introduce dynamic update of alpha and beta parameters with probability 0.2
                        if np.random.rand() < 0.2:
                            self.alpha = max(0.1, self.alpha - 0.001)  # Update alpha
                            self.beta_max = min(1.0, self.beta_max + 0.001)  # Update beta_max

                evaluations += 1
                if evaluations >= self.budget:
                    break

        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution