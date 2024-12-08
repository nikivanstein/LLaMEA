import numpy as np

class HybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 10 * dim
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.best_solution = None
        self.best_fitness = np.inf
        self.current_eval = 0

    def evaluate_population(self, func):
        for i in range(self.pop_size):
            if self.current_eval < self.budget:
                self.fitness[i] = func(self.population[i])
                self.current_eval += 1
                if self.fitness[i] < self.best_fitness:
                    self.best_fitness = self.fitness[i]
                    self.best_solution = self.population[i].copy()

    def mutation(self, idx):
        indices = [i for i in range(self.pop_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        F = 0.5 + np.random.rand() * 0.5
        mutant = np.clip(self.population[a] + F * (self.population[b] - self.population[c]), 
                         self.lower_bound, self.upper_bound)
        return mutant

    def crossover(self, target, mutant):
        crossover_rate = 0.9
        trial = np.copy(target)
        for i in range(self.dim):
            if np.random.rand() < crossover_rate:
                trial[i] = mutant[i]
        return trial

    def annealing(self, candidate, current_temp):
        new_solution = candidate + np.random.normal(0, 0.1, self.dim)
        new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
        new_fitness = float('inf')
        if self.current_eval < self.budget:
            new_fitness = func(new_solution)
            self.current_eval += 1
        if new_fitness < self.best_fitness or np.random.rand() < np.exp((self.best_fitness - new_fitness) / current_temp):
            self.best_fitness = new_fitness
            self.best_solution = new_solution
        return new_solution

    def __call__(self, func):
        self.evaluate_population(func)
        current_temp = 1.0

        while self.current_eval < self.budget:
            for i in range(self.pop_size):
                mutant = self.mutation(i)
                trial = self.crossover(self.population[i], mutant)
                trial_fitness = float('inf')
                if self.current_eval < self.budget:
                    trial_fitness = func(trial)
                    self.current_eval += 1
                
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_solution = trial
                else:
                    self.population[i] = self.annealing(trial, current_temp)

            current_temp *= 0.95

        return self.best_solution