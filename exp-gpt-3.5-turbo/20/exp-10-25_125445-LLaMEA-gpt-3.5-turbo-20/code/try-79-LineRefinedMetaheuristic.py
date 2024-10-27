import numpy as np

class LineRefinedMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.min_val = -5.0
        self.max_val = 5.0
    
    def __call__(self, func):
        population = np.random.uniform(self.min_val, self.max_val, (self.budget, self.dim))
        for _ in range(self.budget):
            selected_index = np.random.randint(self.budget)
            x, a, b, c = population[selected_index], population[np.random.randint(self.budget)], population[np.random.randint(self.budget)], population[np.random.randint(self.budget)]
            if np.random.rand() < 0.2:
                x = np.clip(np.random.normal(x, 0.1), self.min_val, self.max_val)  # Probabilistic line change
            mutant = np.clip(a + np.random.rand() * (b - c), self.min_val, self.max_val)
            trial = np.where(np.random.rand(self.dim) <= 0.5, mutant, x)
            if func(trial) < func(x):
                population[selected_index] = trial.copy()
        return population