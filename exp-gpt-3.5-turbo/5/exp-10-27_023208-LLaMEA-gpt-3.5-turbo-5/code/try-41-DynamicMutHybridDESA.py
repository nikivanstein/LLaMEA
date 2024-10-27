import numpy as np

class DynamicMutHybridDESA(HybridDESA):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.min_mutation_factor = 0.1
        self.max_mutation_factor = 0.9

    def _mutation(self, population, target_index, eval_count):
        mutation_factor = self.min_mutation_factor + (self.max_mutation_factor - self.min_mutation_factor) * eval_count / self.budget
        candidates = [idx for idx in range(len(population)) if idx != target_index]
        a, b, c = np.random.choice(candidates, 3, replace=False)
        mutant = population[a] + mutation_factor * (population[b] - population[c])
        return mutant

    def __call__(self, func):
        population = np.random.uniform(-5, 5, (self.population_size, self.dim))
        eval_count = 0
        best_solution = population[0]
        for _ in range(self.budget):
            for idx, target in enumerate(population):
                mutant = self._mutation(population, idx, eval_count)
                trial = self._crossover(target, mutant)
                eval_count += 1
                energy_diff = func(trial) - func(target)
                if energy_diff < 0 or np.random.rand() < self._acceptance_probability(energy_diff, self.initial_temperature):
                    population[idx] = trial
                    if func(trial) < func(best_solution):
                        best_solution = trial
            self.initial_temperature *= self.alpha
        return best_solution