import numpy as np

class AdaptiveMutation:
    def __init__(self, budget, dim, alpha, beta, epsilon, max_iter):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            if np.random.rand() < self.epsilon:
                new_individual = np.random.choice(self.search_space, size=self.dim, replace=True)
                new_individual = np.clip(new_individual, self.alpha, self.beta)
                new_individual = self.evaluate_fitness(new_individual, func)
            else:
                new_individual = self.search_space
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

def evaluateBBOB(func, budget):
    algorithm = AdaptiveMutation(budget, 10, 0.5, 1.5, 0.1, 1000)
    return algorithm(func)

# Test the algorithm
func = lambda x: x**2
result = evaluateBBOB(func, 1000)
print(result)