import numpy as np
import random

class HybridMetaheuristic:
    def __init__(self, budget, dim, refines=0.7, initial_refine=0.3):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.refined = False
        self.refine_factor = initial_refine
        self.refine_counter = 0

    def __call__(self, func, num_evaluations):
        if self.refined:
            self.refine()
        else:
            self.refine(num_evaluations)

        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
        return func_value

    def refine(self):
        if not self.refined:
            self.refine_factor = random.uniform(0.7, 1.3)
            self.refine_counter = 0
            while self.refine_counter < 10:  # refine strategy: increase the refine factor
                if random.random() < self.refine_factor:
                    self.refine_counter += 1
                    if self.refine_counter >= 5:  # refine strategy: increase the refine factor by 10%
                        self.refine_factor *= 1.1
                else:
                    break
            self.refined = True

def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

hybrid = HybridMetaheuristic(1000, 2)  # 1000 function evaluations, 2 dimensions
print(hybrid(test_function))  # prints a random value between -10 and 10