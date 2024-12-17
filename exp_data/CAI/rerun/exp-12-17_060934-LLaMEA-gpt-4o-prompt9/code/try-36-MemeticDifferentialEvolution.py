import numpy as np

class MemeticDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, int(0.1 * budget))
        self.mutation_factor = 0.85
        self.crossover_rate = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.local_search_prob = 0.15

    def local_search(self, individual, func):
        # Local search using a simple gradient approximation
        step_size = 0.01
        trial = np.copy(individual)
        for i in range(self.dim):
            trial[i] += step_size
            right = func(trial)
            trial[i] -= 2 * step_size
            left = func(trial)
            trial[i] += step_size  # reset to original
            if right < left:
                trial[i] += step_size
            elif left < right:
                trial[i] -= step_size
        return np.clip(trial, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                # Adaptive mutation factor based on diversity
                mutation_factor = self.mutation_factor * ((self.budget - eval_count) / self.budget) * (1 - np.std(fitness) / (np.mean(fitness) + 1e-30))
                mutation_factor *= 1 + 0.1 * (np.std(population, axis=0) / (np.mean(population, axis=0) + 1e-30)).mean()

                mutant_vector = a + mutation_factor * (b - c)
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                self.crossover_rate = 0.65 + 0.35 * (1 - (np.min(fitness) / (np.mean(fitness) + 1e-30)))

                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant_vector, population[i])
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)

                if np.random.rand() < self.local_search_prob:
                    trial_vector = self.local_search(trial_vector, func)

                trial_fitness = func(trial_vector)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness

        return population[np.argmin(fitness)], np.min(fitness)