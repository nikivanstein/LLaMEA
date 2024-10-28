import numpy as np

class AES:
    def __init__(self, budget, dim, alpha=0.01, beta=0.9, gamma=0.01, mu=0.01):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mu = mu
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.best_func = None
        self.best_score = float('-inf')

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
        return func_value

    def adapt(self, func, new_func, new_score):
        if new_score > self.best_score:
            self.best_func = new_func
            self.best_score = new_score
        if new_score - self.best_score > 3 * self.alpha * (self.best_score - self.alpha * self.mu):
            self.alpha = self.alpha * (1 - self.beta)
        elif new_score - self.best_score > 3 * self.alpha * (self.alpha * self.mu - self.beta * self.mu):
            self.beta = self.beta * (1 - self.gamma)
        else:
            self.beta = self.beta * (1 - self.gamma)

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

def evaluate_func(func, x):
    return func(x)

def test_func(x):
    return np.exp(-x[0]**2 - x[1]**2)

aes = AES(1000, 2)  # 1000 function evaluations, 2 dimensions
aes.adapt(test_function, test_func, evaluate_func(test_func, aes.search_space))

print(aes(test_function))  # prints a random value between -10 and 10