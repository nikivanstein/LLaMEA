import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, int(0.1 * budget))
        self.mutation_factor = 0.85
        self.crossover_rate = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        while eval_count < self.budget:
            if eval_count > 0.5 * self.budget:
                self.population_size = max(5, int(0.05 * self.budget))
                population = population[:self.population_size]
                fitness = fitness[:self.population_size]

            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                # Adaptive mutation factor
                mutation_factor = self.mutation_factor * (1 - np.tanh(eval_count / self.budget))

                # Enhanced mutation: weighted difference strategy
                mutant_vector = a + mutation_factor * (b - c) + 0.15 * (population[np.argmin(fitness)] - a)
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                # Adaptive crossover rate
                self.crossover_rate = 0.7 + 0.3 * (1 - np.log(1 + eval_count / self.budget))

                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant_vector, population[i])
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)

                trial_fitness = func(trial_vector)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness

        return population[np.argmin(fitness)], np.min(fitness)