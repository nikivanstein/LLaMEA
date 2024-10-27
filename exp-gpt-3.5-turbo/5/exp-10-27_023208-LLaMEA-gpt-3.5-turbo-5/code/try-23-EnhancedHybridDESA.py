import numpy as np

class EnhancedHybridDESA(HybridDESA):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.temperature_schedule = np.geomspace(self.initial_temperature, self.final_temperature, self.budget)

    def _optimize_func(self, func, population):
        best_solution = population[0]
        for t in range(self.budget):
            for idx, target in enumerate(population):
                mutant = self._mutation(population, idx)
                trial = self._crossover(target, mutant)
                energy_diff = func(trial) - func(target)
                if energy_diff < 0 or np.random.rand() < self._acceptance_probability(energy_diff, self.temperature_schedule[t]):
                    population[idx] = trial
                    if func(trial) < func(best_solution):
                        best_solution = trial
        return best_solution