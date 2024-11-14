import numpy as np

class HPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocity = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, float('inf'))
        self.best_personal_position = np.copy(self.population)
        self.best_personal_fitness = np.full(self.pop_size, float('inf'))
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.evaluations = 0
        self.w = 0.5  # Inertia weight
        self.c1 = 1.5 # Cognitive (personal) component
        self.c2 = 1.5 # Social component
        self.F = 0.8  # Differential Evolution mutation factor
        self.CR = 0.9 # Differential Evolution crossover probability

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def differential_mutation(self, x1, x2, x3):
        return x1 + self.F * (x2 - x3)

    def __call__(self, func):
        np.random.seed(42)

        # Initial evaluation
        for i in range(self.pop_size):
            self.fitness[i] = self.evaluate(func, self.population[i])
            if self.fitness[i] < self.best_personal_fitness[i]:
                self.best_personal_fitness[i] = self.fitness[i]
                self.best_personal_position[i] = self.population[i]
            if self.fitness[i] < self.best_global_fitness:
                self.best_global_fitness = self.fitness[i]
                self.best_global_position = self.population[i]

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                self.velocity[i] = (self.w * self.velocity[i] +
                                    self.c1 * r1 * (self.best_personal_position[i] - self.population[i]) +
                                    self.c2 * r2 * (self.best_global_position - self.population[i]))
                candidate = self.population[i] + self.velocity[i]
                candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                
                indices = [idx for idx in range(self.pop_size) if idx != i]
                x1, x2, x3 = self.population[np.random.choice(indices, 3, replace=False)]
                mutant = self.differential_mutation(x1, x2, x3)
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, candidate)
                trial = np.clip(trial, self.lower_bound, self.upper_bound)

                trial_fitness = self.evaluate(func, trial)
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < self.best_personal_fitness[i]:
                        self.best_personal_fitness[i] = trial_fitness
                        self.best_personal_position[i] = trial
                if trial_fitness < self.best_global_fitness:
                    self.best_global_fitness = trial_fitness
                    self.best_global_position = trial

        return self.best_global_position