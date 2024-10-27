import numpy as np

class AdaptiveLineSearch(HEBBO):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.linesearch = False
        self.boundaries = np.linspace(-5.0, 5.0, dim)

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            if self.linesearch:
                # Adaptive Line Search
                step_size = self.boundaries[func_evaluations % self.budget]
                self.search_space = np.linspace(self.search_space[func_evaluations // self.budget], self.search_space[func_evaluations % self.budget], dim)
            self.linesearch = not self.linesearch
        return func_value

    def mutate(self, individual):
        if np.random.rand() < 0.05:
            # Randomly change an element in the individual
            index = np.random.randint(0, self.dim)
            self.search_space[index] = np.random.uniform(self.search_space[index])
        return individual