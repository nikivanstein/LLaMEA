import numpy as np

class AdvancedChaoticProgressive_Elites:
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
        self.dynamic_temperature = lambda evals: 1.0 - (0.7 * evals / self.budget)  # Reinvented cooling schedule
        self.elite_fraction = 0.15  # Increased elite preservation rate

    def evaluate_population(self, func):
        for i in range(self.population.shape[0]):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
                if self.fitness[i] < self.best_fitness:
                    self.best_fitness = self.fitness[i]
                    self.best_solution = self.population[i].copy()

    def chaos_based_mutation(self, current_index):
        chaotic_factor = np.random.beta(0.6, 0.6)  # Modified chaotic factor
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
        factor = 0.4 + 0.6 * evals / self.budget  # Revised scaling factor
        new_size = int(self.initial_population_size * factor)
        if new_size > self.population.shape[0]:
            additional = new_size - self.population.shape[0]
            new_individuals = np.random.uniform(self.lower_bound, self.upper_bound, (additional, self.dim))
            self.population = np.vstack((self.population, new_individuals))
            self.fitness = np.concatenate((self.fitness, np.inf * np.ones(additional)))

    def elite_preservation(self):
        elite_count = int(self.elite_fraction * self.population.shape[0])
        elite_indices = np.argsort(self.fitness)[:elite_count]
        return self.population[elite_indices], self.fitness[elite_indices]

    def __call__(self, func):
        evals = 0
        self.evaluate_population(func)
        evals += self.population.shape[0]

        while evals < self.budget:
            elites, elite_fitness = self.elite_preservation()
            for i in range(self.population.shape[0]):
                if evals >= self.budget:
                    break
                self.stochastic_tunneling(func, i, evals)
                evals += 1
            self.progressive_population_scaling(evals)

            # Reintroduce elites to maintain diversity
            if self.population.shape[0] > elites.shape[0]:
                self.population[:elites.shape[0]] = elites
                self.fitness[:elites.shape[0]] = elite_fitness

        return self.best_solution, self.best_fitness