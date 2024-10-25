import numpy as np

class HybridGA_DE_ALR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.pop_size = 10 * dim
        self.pop = np.random.uniform(self.lb, self.ub, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, float('inf'))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.func_evals = 0
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.learning_rate = 0.01
    
    def evaluate_population(self, func):
        for i in range(self.pop_size):
            if self.func_evals < self.budget:
                f_value = func(self.pop[i])
                self.func_evals += 1
                self.fitness[i] = f_value
                if f_value < self.best_fitness:
                    self.best_fitness = f_value
                    self.best_solution = self.pop[i]

    def differential_evolution(self, func):
        for i in range(self.pop_size):
            if self.func_evals >= self.budget:
                return
            indices = [idx for idx in range(self.pop_size) if idx != i]
            a, b, c = self.pop[np.random.choice(indices, 3, replace=False)]
            mutant_vector = np.clip(a + self.mutation_factor * (b - c), self.lb, self.ub)
            crossover = np.random.rand(self.dim) < self.crossover_rate
            trial_vector = np.where(crossover, mutant_vector, self.pop[i])
            f_trial = func(trial_vector)
            self.func_evals += 1
            if f_trial < self.fitness[i]:
                self.pop[i] = trial_vector
                self.fitness[i] = f_trial
                if f_trial < self.best_fitness:
                    self.best_fitness = f_trial
                    self.best_solution = trial_vector

    def adaptive_learning_rate(self):
        self.learning_rate = max(0.001, self.learning_rate * (1 - self.func_evals/self.budget))

    def __call__(self, func):
        self.evaluate_population(func)
        while self.func_evals < self.budget:
            self.differential_evolution(func)
            self.adaptive_learning_rate()
            if self.func_evals < self.budget:
                self.pop = np.clip(self.pop + self.learning_rate * np.random.randn(*self.pop.shape), self.lb, self.ub)
                self.evaluate_population(func)
        return self.best_solution