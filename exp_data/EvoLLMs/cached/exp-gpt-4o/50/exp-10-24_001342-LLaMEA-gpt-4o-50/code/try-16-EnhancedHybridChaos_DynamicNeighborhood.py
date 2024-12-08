import numpy as np

class EnhancedHybridChaos_DynamicNeighborhood:
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
        self.dynamic_temperature = lambda evals: 1.0 - (0.85 * evals / self.budget)  # Slightly tweaked cooling
        
    def evaluate_population(self, func):
        for i in range(self.population.shape[0]):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
                if self.fitness[i] < self.best_fitness:
                    self.best_fitness = self.fitness[i]
                    self.best_solution = self.population[i].copy()
    
    def chaos_based_mutation(self, current_index):
        chaotic_factor = np.random.beta(0.6, 0.6)  # Adjusted beta distribution
        a, b = self.population[np.random.choice(self.population.shape[0], 2, replace=False)]
        mutant_vector = np.clip(a + chaotic_factor * (b - self.population[current_index]), self.lower_bound, self.upper_bound)
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
        factor = 0.6 + 0.4 * evals / self.budget  # Modified scaling factor
        new_size = int(self.initial_population_size * factor)
        if new_size > self.population.shape[0]:
            additional = new_size - self.population.shape[0]
            new_individuals = np.random.uniform(self.lower_bound, self.upper_bound, (additional, self.dim))
            self.population = np.vstack((self.population, new_individuals))
            self.fitness = np.concatenate((self.fitness, np.inf * np.ones(additional)))
    
    def dynamic_neighborhood_exploration(self):
        for i in range(self.population.shape[0]):
            if np.random.rand() < 0.3:  # Chance to explore neighborhood
                neighbors = self.population + np.random.normal(0, 0.1, self.population.shape)
                neighbors = np.clip(neighbors, self.lower_bound, self.upper_bound)
                neighbor_fitness = np.array([func(neighbor) for neighbor in neighbors])
                better_neighbors = neighbor_fitness < self.fitness
                self.population[better_neighbors] = neighbors[better_neighbors]
                self.fitness[better_neighbors] = neighbor_fitness[better_neighbors]
    
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
            self.dynamic_neighborhood_exploration()

        return self.best_solution, self.best_fitness