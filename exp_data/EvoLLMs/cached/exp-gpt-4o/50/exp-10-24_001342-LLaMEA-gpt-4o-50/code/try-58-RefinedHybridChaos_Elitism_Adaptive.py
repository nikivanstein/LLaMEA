import numpy as np

class RefinedHybridChaos_Elitism_Adaptive:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 60
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.initial_population_size, self.dim))
        self.fitness = np.inf * np.ones(self.initial_population_size)
        self.best_solution = None
        self.best_fitness = np.inf
        self.dynamic_temperature = lambda evals: 1.0 - (0.8 * evals / self.budget)  # Adjusted cooling schedule
        
    def evaluate_population(self, func):
        for i in range(self.population.shape[0]):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
                if self.fitness[i] < self.best_fitness:
                    self.best_fitness = self.fitness[i]
                    self.best_solution = self.population[i].copy()
    
    def chaos_based_mutation(self, current_index):
        chaotic_factor = np.random.beta(0.7, 0.7)  # Adjusted parameters for chaos
        a, b, c = self.population[np.random.choice(self.population.shape[0], 3, replace=False)]
        mutant_vector = np.clip(a + chaotic_factor * (b - c), self.lower_bound, self.upper_bound)
        return mutant_vector
    
    def stochastic_tunneling(self, func, current_index, evals):
        current_temp = self.dynamic_temperature(evals)
        trial = self.chaos_based_mutation(current_index)
        trial_fitness = func(trial)
        current_fitness = self.fitness[current_index]
        
        if trial_fitness < current_fitness or np.random.rand() < np.exp((current_fitness - trial_fitness) / current_temp):
            self.population[current_index] = trial
            self.fitness[current_index] = trial_fitness
            if trial_fitness < self.best_fitness:
                self.best_fitness = trial_fitness
                self.best_solution = trial.copy()

    def dynamic_elitism(self):
        elite_size = max(1, int(0.1 * self.population.shape[0]))
        elite_idx = np.argsort(self.fitness)[:elite_size]
        self.population = self.population[elite_idx]
        self.fitness = self.fitness[elite_idx]
    
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
            self.dynamic_elitism()
            for i in range(self.population.shape[0]):
                if evals >= self.budget:
                    break
                self.stochastic_tunneling(func, i, evals)
                evals += 1
            self.progressive_population_scaling(evals)

        return self.best_solution, self.best_fitness