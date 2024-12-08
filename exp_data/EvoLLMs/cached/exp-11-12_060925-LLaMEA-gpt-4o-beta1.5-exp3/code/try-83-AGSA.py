import numpy as np

class AGSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.population_size = 10 * dim
        self.temperature = 1.0
        self.cooling_rate = 0.95
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7

    def __call__(self, func):
        population = self.lower_bound + np.random.rand(self.population_size, self.dim) * (self.upper_bound - self.lower_bound)
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations = self.population_size

        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]

        while self.evaluations < self.budget:
            new_population = []

            for _ in range(self.population_size // 2):
                if self.evaluations >= self.budget:
                    break

                parents = self._select_parents(population, fitness)
                offspring1, offspring2 = self._crossover(parents[0], parents[1])

                offspring1 = self._mutate(offspring1)
                offspring2 = self._mutate(offspring2)

                new_population.extend([offspring1, offspring2])

            new_population = np.array(new_population)
            new_fitness = np.apply_along_axis(func, 1, new_population)
            self.evaluations += len(new_population)

            for i in range(len(new_population)):
                if new_fitness[i] < fitness[i] or np.random.rand() < np.exp((fitness[i] - new_fitness[i]) / self.temperature):
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]

                    if new_fitness[i] < best_fitness:
                        best_individual = new_population[i]
                        best_fitness = new_fitness[i]

            self.temperature *= self.cooling_rate

        return best_individual, best_fitness

    def _select_parents(self, population, fitness):
        idx1, idx2 = np.random.choice(len(population), 2, replace=False)
        return (population[idx1], population[idx2]) if fitness[idx1] < fitness[idx2] else (population[idx2], population[idx1])

    def _crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            crossover_point = np.random.randint(1, self.dim - 1)
            offspring1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            offspring2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        else:
            offspring1, offspring2 = parent1.copy(), parent2.copy()
        return offspring1, offspring2

    def _mutate(self, individual):
        for i in range(self.dim):
            if np.random.rand() < self.mutation_rate:
                individual[i] += np.random.randn() * 0.1
                individual[i] = np.clip(individual[i], self.lower_bound, self.upper_bound)
        return individual