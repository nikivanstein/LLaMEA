import numpy as np

class EnhancedHybridChaos_Tunneling_Adaptive:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 60  # Increased initial population size for better diversity
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.initial_population_size, self.dim))
        self.fitness = np.inf * np.ones(self.initial_population_size)
        self.best_solution = None
        self.best_fitness = np.inf
        self.dynamic_temperature = lambda evals: 0.8 - (0.7 * evals / self.budget)  # Modified cooling schedule for faster temperature decrease
        
    def evaluate_population(self, func):
        for i in range(self.population.shape[0]):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
                if self.fitness[i] < self.best_fitness:
                    self.best_fitness = self.fitness[i]
                    self.best_solution = self.population[i].copy()
    
    def chaos_based_mutation(self, current_index):
        chaotic_factor = np.random.beta(0.7, 0.3)  # Adjusted beta distribution for a different mutation pattern
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

    def progressive_population_scaling(self, evals):
        factor = 0.4 + 0.6 * evals / self.budget  # Altered scaling factor for more aggressive scaling
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