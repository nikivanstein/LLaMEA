import numpy as np

class HybridFireflyDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.alpha = 0.5
        self.beta_min = 0.2
        self.beta_max = 1.0
        self.gamma = 0.5
        self.mutation_scale = 0.5
        self.population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.population_size)]
        self.best_solution = min(self.population, key=lambda x: func(x))
        self.best_fitness = func(self.best_solution)

    def __call__(self, func):
        for _ in range(self.budget // self.population_size):
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if func(self.population[j]) < func(self.population[i]):
                        attractiveness = self.beta_min + (self.beta_max - self.beta_min) * np.exp(-self.gamma * np.linalg.norm(self.population[j] - self.population[i]))
                        self.population[i] += attractiveness * (self.population[j] - self.population[i])
                        
            for i in range(self.population_size):
                ind1, ind2, ind3 = np.random.choice(range(self.population_size), 3, replace=False)
                mutant = self.best_solution + self.mutation_scale * (self.population[ind1] - self.population[ind2])
                trial = np.where(np.random.uniform(0, 1, self.dim) < self.alpha, mutant, self.best_solution)
                trial_fitness = func(trial)
                if trial_fitness < self.best_fitness:
                    self.best_solution = trial
                    self.best_fitness = trial_fitness

        return self.best_solution