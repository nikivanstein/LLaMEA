import numpy as np

class HybridFireflyOppositionOptimization:
    def __init__(self, budget, dim, pop_size=30, alpha=0.1, beta0=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

    def __call__(self, func):
        def opp_position(population):
            return 10.0 - population

        def evaluate(population):
            return np.array([func(ind) for ind in population])

        def move_fireflies(firefly, target_firefly):
            r = np.linalg.norm(firefly - target_firefly)
            beta = self.beta0 * np.exp(-self.gamma * r**2)
            new_firefly = firefly + beta * (target_firefly - firefly) + self.alpha * np.random.uniform(-1, 1, self.dim)
            return np.clip(new_firefly, -5.0, 5.0)

        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        op_population = opp_position(population)

        for _ in range(self.budget):
            new_population = np.zeros_like(population)

            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if func(op_population[j]) < func(op_population[i]):
                        population[i] = move_fireflies(population[i], population[j])
                        op_population[i] = opp_position(population[i])

            best_solution = op_population[np.argmin(evaluate(op_population))]

        return best_solution