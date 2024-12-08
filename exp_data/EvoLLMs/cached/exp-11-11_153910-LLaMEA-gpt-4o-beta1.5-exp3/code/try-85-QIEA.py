import numpy as np

class QIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, float('inf'))
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.evaluations = 0
        self.alpha = 0.5  # Quantum rotation angle
        self.prob_amplitudes = np.random.rand(self.pop_size, self.dim)

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def collapse(self):
        collapsed_population = np.where(self.prob_amplitudes > np.random.rand(self.pop_size, self.dim), 
                                        self.upper_bound, self.lower_bound)
        return collapsed_population

    def quantum_rotation(self, idx):
        delta_theta = self.alpha * (self.fitness[idx] - self.best_global_fitness) / np.abs(self.fitness[idx] - self.best_global_fitness + 1e-10)
        rotation_matrix = np.array([[np.cos(delta_theta), -np.sin(delta_theta)], 
                                    [np.sin(delta_theta), np.cos(delta_theta)]])
        self.prob_amplitudes[idx] = np.dot(rotation_matrix, self.prob_amplitudes[idx])

    def __call__(self, func):
        np.random.seed(42)
        
        # Initial evaluation
        for i in range(self.pop_size):
            self.fitness[i] = self.evaluate(func, self.population[i])
            if self.fitness[i] < self.best_global_fitness:
                self.best_global_fitness = self.fitness[i]
                self.best_global_position = self.population[i]

        while self.evaluations < self.budget:
            collapsed_population = self.collapse()

            for i in range(self.pop_size):
                trial = collapsed_population[i]
                trial = np.clip(trial, self.lower_bound, self.upper_bound)
                
                trial_fitness = self.evaluate(func, trial)
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < self.best_global_fitness:
                        self.best_global_fitness = trial_fitness
                        self.best_global_position = trial
                self.quantum_rotation(i)

        return self.best_global_position