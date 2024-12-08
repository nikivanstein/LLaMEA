import numpy as np

class AdaptiveHybridDESA(HybridDESA):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.alpha = 0.95
        self.temperature_schedule = np.geomspace(self.initial_temperature, self.final_temperature, self.budget)

    def _optimize_func(self, func, population):
        best_solution = population[0]
        for t in range(self.budget):
            self.initial_temperature = self.temperature_schedule[t]
            for idx, target in enumerate(population):
                mutant = self._mutation(population, idx)
                trial = self._crossover(target, mutant)
                energy_diff = func(trial) - func(target)
                if energy_diff < 0 or np.random.rand() < self._acceptance_probability(energy_diff, self.initial_temperature):
                    population[idx] = trial
                    if func(trial) < func(best_solution):
                        best_solution = trial
        return best_solution