import numpy as np

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.mutate = False

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            if func_value < 0.5:  # Refine the solution
                self.mutate = True
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

class HESBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.mutate = False
        self.crossover_prob = 0.5
        self.mutation_prob = 0.01

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            if np.random.rand() < self.crossover_prob:  # Perform crossover
                parent1, parent2 = np.random.choice(self.search_space, size=self.dim, replace=False)
                child = (parent1 + parent2) / 2
                self.search_space = np.linspace(-5.0, 5.0, self.dim)
            if np.random.rand() < self.mutation_prob:  # Perform mutation
                if self.mutate:
                    self.search_space = np.random.uniform(-5.0, 5.0, self.dim)
                else:
                    self.search_space = np.linspace(-5.0, 5.0, self.dim)
            self.func_evaluations += 1
        return func_value

class HESBO2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.mutate = False
        self.crossover_prob = 0.5
        self.mutation_prob = 0.01
        self.tournament_size = 2

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            if np.random.rand() < self.crossover_prob:  # Perform crossover
                parent1, parent2 = np.random.choice(self.search_space, size=self.dim, replace=False)
                child = (parent1 + parent2) / 2
                self.search_space = np.linspace(-5.0, 5.0, self.dim)
            if np.random.rand() < self.mutation_prob:  # Perform mutation
                if self.mutate:
                    self.search_space = np.random.uniform(-5.0, 5.0, self.dim)
                else:
                    self.search_space = np.linspace(-5.0, 5.0, self.dim)
            if np.random.rand() < self.tournament_size:  # Perform tournament selection
                tournament = np.random.choice(len(self.search_space), size=self.dim, replace=False)
                tournament_indices = np.argsort(np.random.rand(len(self.search_space), self.dim))
                tournament_indices = tournament_indices[:self.dim]
                tournament_values = np.array([self.search_space[i] for i in tournament_indices])
                tournament_values = np.mean(tournament_values, axis=0)
                self.search_space = tournament_values
            self.func_evaluations += 1
        return func_value

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 