import numpy as np

class EnhancedHybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 15  # Increase population size for better exploration
        self.crossover_rate = 0.8  # Increase crossover rate for more diverse offspring
        self.mutation_factor = 0.6  # Increase mutation factor for broader search
        self.initial_temperature = 1.0
        self.final_temperature = 0.0001
        self.alpha = 0.95  # Increase the cooling rate for faster convergence

    def _optimize_func(self, func, population):
        best_solution = population[0]
        for _ in range(self.budget):
            for idx, target in enumerate(population):
                mutant = self._mutation(population, idx)
                trial = self._crossover(target, mutant)
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