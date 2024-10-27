import numpy as np

class AES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.p = 0.95
        self.d = 0.01
        self.c1 = 2
        self.c2 = 2
        self.c3 = 2

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            if func_evaluations > self.budget:
                raise ValueError("Exceeded budget of function evaluations")

            # Adaptive mutation strategy
            if np.random.rand() < self.p:
                new_individual = np.copy(self.search_space)
                new_individual[np.random.randint(0, self.dim)] = np.random.uniform(-5.0, 5.0)
                new_individual = np.clip(new_individual, -5.0, 5.0)
                self.search_space = np.linspace(-5.0, 5.0, self.dim)

            # Adaptive crossover strategy
            if np.random.rand() < self.c1:
                parent1, parent2 = np.random.choice(self.search_space, size=self.dim, replace=False)
                child = (parent1 + parent2) / 2
            else:
                child = np.random.choice(self.search_space, size=self.dim, replace=False)

            # Adaptive mutation strategy
            if np.random.rand() < self.c2:
                new_child = np.copy(child)
                if np.random.rand() < self.d:
                    new_child[np.random.randint(0, self.dim)] = np.random.uniform(-5.0, 5.0)
                new_child = np.clip(new_child, -5.0, 5.0)
                self.search_space = np.linspace(-5.0, 5.0, self.dim)

            # Evaluate new individual
            self.func_evaluations += 1
            new_individual = self.evaluate_fitness(new_individual, func)

            # Update population
            if new_individual < func:
                self.search_space = np.linspace(-5.0, 5.0, self.dim)
            else:
                self.search_space = np.linspace(-5.0, 5.0, self.dim)

    def evaluate_fitness(self, individual, func):
        func_value = func(individual)
        return func_value