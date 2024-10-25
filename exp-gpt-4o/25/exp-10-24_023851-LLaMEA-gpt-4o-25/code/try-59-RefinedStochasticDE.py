import numpy as np

class RefinedStochasticDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.bounds = (-5.0, 5.0)
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.evaluations = 0
        self.adaptive_mutation_factor = 0.7
        self.success_rate = 0.1

    def __call__(self, func):
        def stochastic_differential_evolution(population):
            new_population = []
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 5, replace=False)
                a, b, c, d = population[indices[0]], population[indices[1]], population[indices[2]], population[indices[3]]

                # Using a stochastic approach for the mutation factor
                self.mutation_factor = np.random.uniform(0.5, self.adaptive_mutation_factor)

                mutant = np.clip(a + self.mutation_factor * (b - c) + 0.3 * (d - b), *self.bounds)

                trial = np.copy(population[i])
                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial[crossover] = mutant[crossover]

                if func(trial) < func(population[i]):
                    new_population.append(trial)
                    self.success_rate = min(1.0, self.success_rate + 0.03)
                else:
                    new_population.append(population[i])
                    self.success_rate = max(0.01, self.success_rate - 0.01)

                self.evaluations += 1
                if self.evaluations >= self.budget:
                    return new_population
            return new_population

        def adaptive_local_search(individual):
            dynamic_radius = (self.bounds[1] - self.bounds[0]) * 0.01 * (1 - self.success_rate)
            best_local = np.copy(individual)
            best_val = func(best_local)

            for _ in range(20):
                perturbation = np.random.normal(0, dynamic_radius, self.dim)
                candidate = np.clip(best_local + perturbation, *self.bounds)

                candidate_val = func(candidate)
                self.evaluations += 1
                if candidate_val < best_val:
                    best_local, best_val = candidate, candidate_val

                if self.evaluations >= self.budget:
                    break

            return best_local

        population = np.random.uniform(*self.bounds, (self.population_size, self.dim))

        while self.evaluations < self.budget:
            population = stochastic_differential_evolution(population)

            # Apply local search dynamically on selected top individuals
            if self.evaluations < self.budget * 0.8:
                population = sorted(population, key=func)
                best_count = max(4, self.population_size // 10)
                for i in range(min(best_count, len(population))):
                    population[i] = adaptive_local_search(population[i])
                    if self.evaluations >= self.budget:
                        break

        best_individual = min(population, key=func)
        return best_individual