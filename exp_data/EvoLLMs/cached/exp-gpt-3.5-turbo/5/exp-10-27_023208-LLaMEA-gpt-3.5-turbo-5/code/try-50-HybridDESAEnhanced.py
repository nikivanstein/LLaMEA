import numpy as np

class HybridDESAEnhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.crossover_rate = 0.7
        self.mutation_factor = 0.5
        self.initial_temperature = 1.0
        self.final_temperature = 0.0001
        self.alpha = 0.9

    def _mutation(self, population, target_index):
        candidates = [idx for idx in range(len(population)) if idx != target_index]
        a, b, c = np.random.choice(candidates, 3, replace=False)
        mutant = population[a] + self.mutation_factor * (population[b] - population[c])
        return mutant

    def _crossover(self, target_vector, mutant_vector):
        crossover_points = np.random.rand(self.dim) < self.crossover_rate
        trial_vector = np.where(crossover_points, mutant_vector, target_vector)
        return trial_vector

    def _hill_climber(self, func, individual):
        current_solution = individual
        for _ in range(self.budget // 10):  # Hill climber budget set to 10% of total budget
            candidate = current_solution + np.random.normal(0, 0.1, self.dim)
            if func(candidate) < func(current_solution):
                current_solution = candidate
        return current_solution

    def _acceptance_probability(self, energy_diff, temperature):
        return np.exp(-energy_diff / temperature)

    def _optimize_func(self, func, population):
        best_solution = population[0]
        for _ in range(self.budget):
            for idx, target in enumerate(population):
                mutant = self._mutation(population, idx)
                trial = self._crossover(target, mutant)
                trial = self._hill_climber(func, trial)  # Integrate hill climber
                energy_diff = func(trial) - func(target)
                if energy_diff < 0 or np.random.rand() < self._acceptance_probability(energy_diff, self.initial_temperature):
                    population[idx] = trial
                    if func(trial) < func(best_solution):
                        best_solution = trial
            self.initial_temperature *= self.alpha
        return best_solution

    def __call__(self, func):
        population = np.random.uniform(-5, 5, (self.population_size, self.dim))
        return self._optimize_func(func, population)