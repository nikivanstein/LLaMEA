import numpy as np

class AdaptiveDE(EvolutionaryAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.prob_mutation = 0.5

    def _mutate(self, population, target_index):
        candidates = population[np.arange(self.pop_size) != target_index]
        a, b, c = np.random.choice(candidates, 3, replace=False)
        
        if np.random.rand() < self.prob_mutation:
            return np.clip(a + self.f * (b - c), self.lower_bound, self.upper_bound)
        else:
            return population[target_index]

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

            # Adaptive probability update
            success_rate = sum([func(individual) < target_fitness for individual in population]) / self.pop_size
            self.prob_mutation = 0.35 + 0.6 * success_rate

        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution