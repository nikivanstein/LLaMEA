import numpy as np

class AdaptiveNicheBasedMultiPhaseDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population_size = self.initial_population_size
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = population_size
        elite_fraction = 0.2

        while evaluations < self.budget:
            niche_radius = 0.01 * (1 - evaluations / self.budget) * (self.upper_bound - self.lower_bound)
            mutation_factor = 0.5 + 0.5 * (1 - evaluations / self.budget)
            crossover_rate = 0.7 + 0.3 * (evaluations / self.budget)

            new_population = np.copy(population)

            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                # Adaptive niche control
                if np.linalg.norm(a - b) < niche_radius and np.linalg.norm(b - c) < niche_radius:
                    mutation_factor *= 0.5  # Increase diversity if in same niche

                mutant = np.clip(a + mutation_factor * (b - c), self.lower_bound, self.upper_bound)

                trial = np.array([
                    mutant[j] if np.random.rand() < crossover_rate or j == np.random.randint(self.dim) else population[i][j]
                    for j in range(self.dim)
                ])

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

            if evaluations < self.budget and evaluations > self.budget * 0.3:
                elite_size = max(1, int(elite_fraction * population_size))
                sorted_indices = np.argsort(fitness)
                elite_population = population[sorted_indices[:elite_size]]
                new_population_size = max(4, int(self.initial_population_size * (1 - evaluations/self.budget)))
                new_population = np.vstack((new_population[sorted_indices[:new_population_size - elite_size]], elite_population))
                fitness = fitness[sorted_indices[:new_population_size]]
                population_size = new_population_size

            population = new_population

        return best_solution