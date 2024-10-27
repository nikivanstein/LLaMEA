import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.c1 = 2.05
        self.c2 = 2.05
        self.cr = 0.5
        self.f = 0.5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))

    def _mutate(self, population, target_index):
        candidates = population[np.arange(self.pop_size) != target_index]
        a, b, c = np.random.choice(candidates, 3, replace=False)
        return np.clip(a + self.f * (b - c), self.lower_bound, self.upper_bound)

    def _crossover(self, target_vector, mutant_vector):
        crossover_points = np.random.rand(self.dim) < self.cr
        trial_vector = np.where(crossover_points, mutant_vector, target_vector)
        return trial_vector

    def _update_velocity(self, velocity, position, global_best):
        r1, r2 = np.random.rand(2)
        cognitive = self.c1 * r1 * (position - position)
        social = self.c2 * r2 * (global_best - position)
        return velocity + cognitive + social

    def __call__(self, func):
        population = self._initialize_population()
        velocities = np.zeros((self.pop_size, self.dim))
        global_best = population[np.argmin([func(individual) for individual in population])]
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

                velocities[i] = self._update_velocity(velocities[i], population[i], global_best)
                population[i] = np.clip(population[i] + velocities[i], self.lower_bound, self.upper_bound)

                if evals >= self.budget:
                    break

        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution