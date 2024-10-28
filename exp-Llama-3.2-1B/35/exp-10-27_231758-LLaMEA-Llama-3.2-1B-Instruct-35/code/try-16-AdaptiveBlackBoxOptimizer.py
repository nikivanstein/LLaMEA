import numpy as np
import random
import time
from scipy.optimize import differential_evolution

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None

    def __call__(self, func):
        if self.func_values is None:
            start_time = time.time()
            self.func_evals = self.budget
            self.func_values = np.zeros(self.dim)
            for _ in range(self.func_evals):
                func(self.func_values)
            end_time = time.time()
            print(f"Function evaluations: {self.func_evals}, Time taken: {end_time - start_time} seconds")
        else:
            while self.func_evals > 0:
                idx = np.argmin(np.abs(self.func_values))
                self.func_values[idx] = func(self.func_values[idx])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break

    def adaptive_search(self, func, bounds, initial_guess):
        while self.func_evals > 0:
            idx = np.argmin(np.abs(initial_guess))
            new_guess = initial_guess + random.uniform(-bounds[idx], bounds[idx])
            new_guess = np.clip(new_guess, bounds[idx], bounds[idx])
            func(new_guess)
            self.func_evals -= 1
            if self.func_evals == 0:
                break
        return new_guess

    def adaptive_black_box(self, func, bounds, initial_guess, num_iterations):
        for _ in range(num_iterations):
            new_guess = self.adaptive_search(func, bounds, initial_guess)
            self.func_values = np.copy(new_guess)
            self.func_evals = num_iterations
            if self.func_evals == 0:
                break
        return self.func_values

    def adaptive_bounded_search(self, func, bounds, initial_guess, max_evals):
        for _ in range(max_evals):
            idx = np.argmin(np.abs(initial_guess))
            new_guess = initial_guess + random.uniform(-bounds[idx], bounds[idx])
            new_guess = np.clip(new_guess, bounds[idx], bounds[idx])
            func(new_guess)
            if self.func_evals > 0:
                break
        return new_guess

    def adaptive_random_search(self, func, bounds, initial_guess, num_iterations):
        for _ in range(num_iterations):
            new_guess = initial_guess + random.uniform(-bounds[0], bounds[0])
            new_guess = np.clip(new_guess, bounds[0], bounds[0])
            func(new_guess)
            if self.func_evals > 0:
                break
        return new_guess