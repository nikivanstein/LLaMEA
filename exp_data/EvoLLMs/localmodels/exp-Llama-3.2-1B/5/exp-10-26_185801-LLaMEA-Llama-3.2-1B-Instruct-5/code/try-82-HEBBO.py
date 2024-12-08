import numpy as np

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.iteration = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
            if func_evaluations % 100 == 0:
                self.iteration += 1
                print(f"Iteration {self.iteration}: {func_value:.4f} evaluations made")

        return func_value

class HEBBOHybrid:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.iteration = 0
        self.iteration_strategy = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
            if func_evaluations % 100 == 0:
                self.iteration += 1
                print(f"Iteration {self.iteration}: {func_value:.4f} evaluations made")

            # Novel Hybrid Algorithm
            if self.iteration_strategy == 0:
                # Line Search
                step_size = 0.01
                new_individual = self.evaluate_fitness(self.func(self.search_space))
                new_individual = self.search_space + step_size * (new_individual - self.search_space)
                self.search_space = new_individual
            elif self.iteration_strategy == 1:
                # Crossover
                parent1, parent2 = self.evaluate_fitness(self.func(self.search_space))
                child = (parent1 + parent2) / 2
                self.search_space = np.linspace(-5.0, 5.0, self.dim)
                self.func_evaluations += 1
                print(f"Iteration {self.iteration}: Child {func_value:.4f} evaluations made")
            elif self.iteration_strategy == 2:
                # Mutation
                mutation_rate = 0.1
                if np.random.rand() < mutation_rate:
                    self.search_space = np.random.uniform(-5.0, 5.0, self.dim)
                    print(f"Mutation: New individual {func_value:.4f} evaluations made")

        return func_value

def evaluateBBOB(func, search_space, budget):
    algorithm = HEBBO(budget, len(search_space))
    return algorithm(func)

# Example usage:
func = lambda x: x**2
search_space = np.linspace(-5.0, 5.0, 10)
result = evaluateBBOB(func, search_space, 1000)
print(result)