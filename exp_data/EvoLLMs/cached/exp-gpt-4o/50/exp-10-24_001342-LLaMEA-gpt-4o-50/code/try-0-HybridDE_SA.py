import numpy as np

class HybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.inf * np.ones(self.population_size)
        self.best_solution = None
        self.best_fitness = np.inf
        self.temperatures = np.linspace(1.0, 0.01, budget)

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
                if self.fitness[i] < self.best_fitness:
                    self.best_fitness = self.fitness[i]
                    self.best_solution = self.population[i].copy()

    def differential_evolution(self, current_index):
        idxs = [idx for idx in range(self.population_size) if idx != current_index]
        a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
        mutant_vector = np.clip(a + 0.8 * (b - c), self.lower_bound, self.upper_bound)
        crossover = np.random.rand(self.dim) < 0.9
        trial_vector = np.where(crossover, mutant_vector, self.population[current_index])
        return trial_vector

    def simulated_annealing(self, trial, current_fitness, index, current_temp):
        trial_fitness = func(trial)
        if trial_fitness < current_fitness or np.random.rand() < np.exp((current_fitness - trial_fitness) / current_temp):
            self.population[index] = trial
            self.fitness[index] = trial_fitness
            if trial_fitness < self.best_fitness:
                self.best_fitness = trial_fitness
                self.best_solution = trial.copy()

    def __call__(self, func):
        evals = 0
        self.evaluate_population(func)
        evals += self.population_size

        while evals < self.budget:
            for i in range(self.population_size):
                if evals >= self.budget:
                    break
                current_temp = self.temperatures[evals]
                trial_vector = self.differential_evolution(i)
                self.simulated_annealing(trial_vector, self.fitness[i], i, current_temp)
                evals += 1

        return self.best_solution, self.best_fitness