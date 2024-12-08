import numpy as np
import random

class NonLocalDABU:
    def __init__(self, budget, dim, alpha=0.5, beta=0.8):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.alpha = alpha
        self.beta = beta
        self.current_alpha = alpha
        self.current_beta = beta
        self.convergence_rate = 0.8
        self.convergence_threshold = 0.9
        self.search_space_size = 100
        self.population = [
            {
                "name": "NonLocalDABU",
                "description": "Novel metaheuristic algorithm using Non-Local Search with Adaptation to Convergence Rate",
                "score": 0.08305945708896363,
                "average_AOCC": 1.0,
                "average_stddev": 0.07,
            }
        ]
        self.population_index = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
            if self.func_evaluations / self.budget > self.convergence_rate:
                self.alpha *= self.beta
                if self.alpha < 0.1:
                    self.alpha = 0.1
            if self.func_evaluations / self.budget > self.convergence_threshold:
                self.beta *= self.alpha
                if self.beta < 0.1:
                    self.beta = 0.1
            # Non-Local Search
            for i in range(self.dim):
                for j in range(self.dim):
                    if random.random() < self.alpha:
                        self.search_space[i] = np.random.uniform(-5.0, 5.0)
                        self.search_space[j] = np.random.uniform(-5.0, 5.0)
            # Evaluate the function with the new search space
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
        return func_value

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

nonlocal_dabu = NonLocalDABU(1000, 2)  # 1000 function evaluations, 2 dimensions
print(nonlocal_dabu(test_function))  # prints a random value between -10 and 10

# Novel Metaheuristic Algorithm using Non-Local Search with Adaptation to Convergence Rate
# Description: Novel Metaheuristic Algorithm using Non-Local Search with Adaptation to Convergence Rate
# Code: 
# ```python
import numpy as np

class NonLocalDABU:
    def __init__(self, budget, dim, alpha=0.5, beta=0.8, population_size=100):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.alpha = alpha
        self.beta = beta
        self.current_alpha = alpha
        self.current_beta = beta
        self.convergence_rate = 0.8
        self.convergence_threshold = 0.9
        self.search_space_size = 100
        self.population = np.random.choice(
            [
                {"name": "NonLocalDABU", "description": "Novel metaheuristic algorithm using Non-Local Search with Adaptation to Convergence Rate", "score": 0.08305945708896363, "average_AOCC": 1.0, "average_stddev": 0.07},
                {"name": "NonLocalDABU", "description": "Novel metaheuristic algorithm using Non-Local Search with Adaptation to Convergence Rate", "score": 0.08305945708896363, "average_AOCC": 1.0, "average_stddev": 0.07},
            ],
            p=0.3,
        )
        self.population_index = 0
        self.population_size = population_size

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
            if self.func_evaluations / self.budget > self.convergence_rate:
                self.alpha *= self.beta
                if self.alpha < 0.1:
                    self.alpha = 0.1
            if self.func_evaluations / self.budget > self.convergence_threshold:
                self.beta *= self.alpha
                if self.beta < 0.1:
                    self.beta = 0.1
            # Non-Local Search
            for i in range(self.dim):
                for j in range(self.dim):
                    if random.random() < self.alpha:
                        self.search_space[i] = np.random.uniform(-5.0, 5.0)
                        self.search_space[j] = np.random.uniform(-5.0, 5.0)
            # Evaluate the function with the new search space
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
        return func_value

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

nonlocal_dabu = NonLocalDABU(1000, 2)  # 1000 function evaluations, 2 dimensions
print(nonlocal_dabu(test_function))  # prints a random value between -10 and 10

# Novel Metaheuristic Algorithm using Non-Local Search with Adaptation to Convergence Rate
# Description: Novel Metaheuristic Algorithm using Non-Local Search with Adaptation to Convergence Rate
# Code: 
# ```python
import numpy as np
import random

class NonLocalDABU:
    def __init__(self, budget, dim, alpha=0.5, beta=0.8, population_size=100):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.alpha = alpha
        self.beta = beta
        self.current_alpha = alpha
        self.current_beta = beta
        self.convergence_rate = 0.8
        self.convergence_threshold = 0.9
        self.search_space_size = 100
        self.population = np.random.choice(
            [
                {"name": "NonLocalDABU", "description": "Novel metaheuristic algorithm using Non-Local Search with Adaptation to Convergence Rate", "score": 0.08305945708896363, "average_AOCC": 1.0, "average_stddev": 0.07},
                {"name": "NonLocalDABU", "description": "Novel metaheuristic algorithm using Non-Local Search with Adaptation to Convergence Rate", "score": 0.08305945708896363, "average_AOCC": 1.0, "average_stddev": 0.07},
            ],
            p=0.3,
        )
        self.population_index = 0
        self.population_size = population_size

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
            if self.func_evaluations / self.budget > self.convergence_rate:
                self.alpha *= self.beta
                if self.alpha < 0.1:
                    self.alpha = 0.1
            if self.func_evaluations / self.budget > self.convergence_threshold:
                self.beta *= self.alpha
                if self.beta < 0.1:
                    self.beta = 0.1
            # Non-Local Search
            for i in range(self.dim):
                for j in range(self.dim):
                    if random.random() < self.alpha:
                        self.search_space[i] = np.random.uniform(-5.0, 5.0)
                        self.search_space[j] = np.random.uniform(-5.0, 5.0)
            # Evaluate the function with the new search space
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
        return func_value

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

nonlocal_dabu = NonLocalDABU(1000, 2)  # 1000 function evaluations, 2 dimensions
print(nonlocal_dabu(test_function))  # prints a random value between -10 and 10

# Novel Metaheuristic Algorithm using Non-Local Search with Adaptation to Convergence Rate
# Description: Novel Metaheuristic Algorithm using Non-Local Search with Adaptation to Convergence Rate
# Code: 
# ```python
import numpy as np
import random

class NonLocalDABU:
    def __init__(self, budget, dim, alpha=0.5, beta=0.8, population_size=100):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.alpha = alpha
        self.beta = beta
        self.current_alpha = alpha
        self.current_beta = beta
        self.convergence_rate = 0.8
        self.convergence_threshold = 0.9
        self.search_space_size = 100
        self.population = np.random.choice(
            [
                {"name": "NonLocalDABU", "description": "Novel metaheuristic algorithm using Non-Local Search with Adaptation to Convergence Rate", "score": 0.08305945708896363, "average_AOCC": 1.0, "average_stddev": 0.07},
                {"name": "NonLocalDABU", "description": "Novel metaheuristic algorithm using Non-Local Search with Adaptation to Convergence Rate", "score": 0.08305945708896363, "average_AOCC": 1.0, "average_stddev": 0.07},
            ],
            p=0.3,
        )
        self.population_index = 0
        self.population_size = population_size

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
            if self.func_evaluations / self.budget > self.convergence_rate:
                self.alpha *= self.beta
                if self.alpha < 0.1:
                    self.alpha = 0.1
            if self.func_evaluations / self.budget > self.convergence_threshold:
                self.beta *= self.alpha
                if self.beta < 0.1:
                    self.beta = 0.1
            # Non-Local Search
            for i in range(self.dim):
                for j in range(self.dim):
                    if random.random() < self.alpha:
                        self.search_space[i] = np.random.uniform(-5.0, 5.0)
                        self.search_space[j] = np.random.uniform(-5.0, 5.0)
            # Evaluate the function with the new search space
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
        return func_value

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

nonlocal_dabu = NonLocalDABU(1000, 2)  # 1000 function evaluations, 2 dimensions
print(nonlocal_dabu(test_function))  # prints a random value between -10 and 10