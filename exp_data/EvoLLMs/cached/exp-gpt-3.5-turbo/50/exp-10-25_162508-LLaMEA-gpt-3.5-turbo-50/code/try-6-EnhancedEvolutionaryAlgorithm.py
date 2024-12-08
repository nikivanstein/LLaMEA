import numpy as np

class EnhancedEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.cr = 0.5
        self.f = 0.5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))

    def _mutate(self, population, target_index):
        candidates = population[np.arange(self.pop_size) != target_index]
        a, b, c = np.random.choice(candidates, 3, replace=False)
        line_prob = np.random.rand(self.dim) < 0.35
        mutated_vector = np.clip(population[target_index] + self.f * (b - c), self.lower_bound, self.upper_bound)
        return np.where(line_prob, mutated_vector, population[target_index])

    def _crossover(self, target_vector, mutant_vector):
        crossover_points = np.random.rand(self.dim) < self.cr
        trial_vector = np.where(crossover_points, mutant_vector, target_vector)
        return trial_vector

    def __call__(self, func):
        population = self._initialize_population()
        evals = 0

        while evals < self.budget:
            for i in range(self.pop_size):
                target_vector = population[i]
                mutant_vector = self._mutate(population, i)
                trial_vector = self._crossover(target_vector, mutant_vector)

                target_fitness = func(target_vector)
                trial_fitness = func(trial_vector)
                evals += 1

                if trial_fitness < target_fitness:
                    population[i] = trial_vector

                if evals >= self.budget:
                    break

        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution