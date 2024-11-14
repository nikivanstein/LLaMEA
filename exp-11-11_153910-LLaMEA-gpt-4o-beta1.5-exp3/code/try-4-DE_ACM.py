import numpy as np

class DE_ACM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.evaluations = 0

    def evaluate(self, func, solution):
        fitness = func(solution)
        self.evaluations += 1
        return fitness

    def __call__(self, func):
        np.random.seed(42)
        F_min, F_max = 0.4, 0.9
        CR_min, CR_max = 0.1, 0.9

        while self.evaluations < self.budget:
            F = F_min + (F_max - F_min) * np.random.rand()
            CR = CR_min + (CR_max - CR_min) * np.random.rand()
            
            for i in range(self.pop_size):
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                mutant_vector = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
                
                trial_vector = np.copy(self.population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial_vector[j] = mutant_vector[j]

                trial_fitness = self.evaluate(func, trial_vector)

                if trial_fitness < self.best_global_fitness:
                    self.best_global_fitness = trial_fitness
                    self.best_global_position = trial_vector

                if trial_fitness < self.evaluate(func, self.population[i]):
                    self.population[i] = trial_vector

        return self.best_global_position