import numpy as np

class EnhancedHybridLevy_AdaptiveCrossover:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 50
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.initial_population_size, self.dim))
        self.fitness = np.inf * np.ones(self.initial_population_size)
        self.best_solution = None
        self.best_fitness = np.inf
        self.dynamic_temperature = lambda evals: 1.0 - (0.8 * evals / self.budget)
        
    def evaluate_population(self, func):
        for i in range(self.population.shape[0]):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
                if self.fitness[i] < self.best_fitness:
                    self.best_fitness = self.fitness[i]
                    self.best_solution = self.population[i].copy()
    
    def levy_flight_mutation(self, current_index):
        beta = 1.5
        sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / abs(v) ** (1 / beta)
        mutant_vector = np.clip(self.population[current_index] + 0.01 * step, self.lower_bound, self.upper_bound)
        return mutant_vector
    
    def stochastic_tunneling(self, func, current_index, evals):
        current_temp = self.dynamic_temperature(evals)
        trial = self.levy_flight_mutation(current_index)
        trial_fitness = func(trial)
        current_fitness = self.fitness[current_index]
        
        if trial_fitness < current_fitness or np.random.rand() < np.exp((current_fitness - trial_fitness) / current_temp):
            self.population[current_index] = trial
            self.fitness[current_index] = trial_fitness
            if trial_fitness < self.best_fitness:
                self.best_fitness = trial_fitness
                self.best_solution = trial.copy()

    def progressive_population_scaling(self, evals):
        factor = 0.6 + 0.4 * evals / self.budget
        new_size = int(self.initial_population_size * factor)
        if new_size > self.population.shape[0]:
            additional = new_size - self.population.shape[0]
            new_individuals = np.random.uniform(self.lower_bound, self.upper_bound, (additional, self.dim))
            self.population = np.vstack((self.population, new_individuals))
            self.fitness = np.concatenate((self.fitness, np.inf * np.ones(additional)))
    
    def __call__(self, func):
        evals = 0
        self.evaluate_population(func)
        evals += self.population.shape[0]

        while evals < self.budget:
            for i in range(self.population.shape[0]):
                if evals >= self.budget:
                    break
                self.stochastic_tunneling(func, i, evals)
                evals += 1
            self.progressive_population_scaling(evals)

        return self.best_solution, self.best_fitness