import numpy as np

class HybridFireflyDE:
    def __init__(self, budget, dim, firefly_population=20, mutation_rate=0.5, alpha=0.5, beta_min=0.2, beta_max=1.0):
        self.budget = budget
        self.dim = dim
        self.firefly_population = firefly_population
        self.mutation_rate = mutation_rate
        self.alpha = alpha
        self.beta_min = beta_min
        self.beta_max = beta_max

    def __call__(self, func):
        def evaluate_solution(solution):
            return func(solution)

        def initialize_fireflies():
            return np.random.uniform(-5.0, 5.0, (self.firefly_population, self.dim))

        def move_fireflies(fireflies):
            new_fireflies = []
            for idx, firefly in enumerate(fireflies):
                for other_idx, other_firefly in enumerate(fireflies):
                    if evaluate_solution(other_firefly) < evaluate_solution(firefly):
                        distance = np.linalg.norm(firefly - other_firefly)
                        beta = self.beta_min + (self.beta_max - self.beta_min) * np.exp(-self.alpha * distance)
                        new_position = firefly + beta * (other_firefly - firefly) + self.mutation_rate * np.random.randn(self.dim)
                        new_fireflies.append(new_position)
                    else:
                        new_fireflies.append(firefly + self.mutation_rate * np.random.randn(self.dim))

            return np.array(new_fireflies)

        best_solution = None
        best_fitness = np.inf

        fireflies = initialize_fireflies()
        for _ in range(self.budget // self.firefly_population):
            fireflies = move_fireflies(fireflies)
            for firefly in fireflies:
                fitness = evaluate_solution(firefly)
                if fitness < best_fitness:
                    best_solution = firefly
                    best_fitness = fitness

        return best_solution