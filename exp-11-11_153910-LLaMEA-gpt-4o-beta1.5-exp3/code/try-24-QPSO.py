import numpy as np

class QPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, float('inf'))
        self.best_personal_position = np.copy(self.population)
        self.best_personal_fitness = np.full(self.pop_size, float('inf'))
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.evaluations = 0
        self.alpha = 0.75  # control parameter for quantum behavior

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def quantum_update(self, particle, global_best):
        p = np.random.rand(self.dim)
        u = np.random.rand(self.dim)
        mbest = np.mean(self.best_personal_position, axis=0)
        delta = np.abs(particle - mbest)
        quantum_particle = mbest + np.sign(u - 0.5) * delta * np.log(1 / p)
        return np.clip(quantum_particle, self.lower_bound, self.upper_bound)
    
    def __call__(self, func):
        np.random.seed(42)
        
        # Initial evaluation
        for i in range(self.pop_size):
            self.fitness[i] = self.evaluate(func, self.population[i])
            self.best_personal_fitness[i] = self.fitness[i]
            if self.fitness[i] < self.best_global_fitness:
                self.best_global_fitness = self.fitness[i]
                self.best_global_position = self.population[i]

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                quantum_particle = self.quantum_update(self.population[i], self.best_global_position)
                trial_fitness = self.evaluate(func, quantum_particle)

                if trial_fitness < self.fitness[i]:
                    self.population[i] = quantum_particle
                    self.fitness[i] = trial_fitness
                    if trial_fitness < self.best_personal_fitness[i]:
                        self.best_personal_fitness[i] = trial_fitness
                        self.best_personal_position[i] = quantum_particle
                    if trial_fitness < self.best_global_fitness:
                        self.best_global_fitness = trial_fitness
                        self.best_global_position = quantum_particle

        return self.best_global_position