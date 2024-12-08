import numpy as np
import skfuzzy as fuzz

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(4, 5 + int(self.dim * np.log(self.dim)))
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.eval_count = 0
        self.best_fitness = np.inf

    def fuzzy_control(self, fitness):
        # Fuzzy logic control for mutation and crossover rates
        rates = np.linspace(0, 1, self.population_size)
        low = fuzz.trimf(rates, [0, 0, 0.5])
        medium = fuzz.trimf(rates, [0, 0.5, 1])
        high = fuzz.trimf(rates, [0.5, 1, 1])
        membership = fuzz.interp_membership(rates, high, fitness)
        return membership * 0.9 + (1 - membership) * 0.1  # Fuzzy membership to influence rate

    def __call__(self, func):
        self.evaluate_population(func)
        best_index = np.argmin(self.fitness)
        best_solution = self.population[best_index].copy()
        self.best_fitness = self.fitness[best_index]

        while self.eval_count < self.budget:
            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break

                mutation_factor = self.fuzzy_control(self.fitness[i])
                crossover_rate = self.fuzzy_control(self.fitness[i])

                # Mutation with fuzzy controlled perturbation
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = self.population[indices]
                perturbation = np.random.randn(self.dim) * 0.1 * (best_solution - x0)
                mutant_vector = x0 + mutation_factor * (x1 - x2) + perturbation
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                # Crossover with fuzzy controlled rate
                trial_vector = np.where(np.random.rand(self.dim) < crossover_rate, mutant_vector, self.population[i])

                # Selection
                trial_fitness = func(trial_vector)
                self.eval_count += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial_vector
                    self.fitness[i] = trial_fitness

                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        best_solution = trial_vector.copy()

        return best_solution

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.eval_count >= self.budget:
                break
            self.fitness[i] = func(self.population[i])
            self.eval_count += 1