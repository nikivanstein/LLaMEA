import numpy as np

class CrowdedDE(AdaptiveDE):
    def __init__(self, budget, dim, F=0.8, CR=0.9, pop_size=20):
        super().__init__(budget, dim, F, CR, pop_size)

    def __call__(self, func):
        def crowding_distance(population, fitness):
            dist = np.zeros(len(population))
            sorted_indices = np.argsort(fitness)
            dist[sorted_indices[0]] = dist[sorted_indices[-1]] = np.inf
            for i in range(1, len(population) - 1):
                dist[sorted_indices[i]] += fitness[sorted_indices[i + 1]] - fitness[sorted_indices[i - 1]]
            return dist

        def select_parents(population, fitness, num_parents):
            crowd_dist = crowding_distance(population, fitness)
            return population[np.argsort(crowd_dist)][-num_parents:]

        population = np.random.uniform(-5, 5, (self.pop_size, self.dim))
        fitness = np.array([func(individual) for individual in population])

        for _ in range(self.budget):
            adapt_F = self.F * (1.0 - _ / self.budget)
            adapt_CR = self.CR + 0.1 * np.sin(0.9 * np.pi * _ / self.budget)
            new_population = []
            for i, target in enumerate(population):
                parents = select_parents(population, fitness, 3)
                mutant = mutate(target, parents, adapt_F)
                trial = crossover(target, mutant, adapt_CR)
                new_fitness = func(trial)
                if new_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = new_fitness
                new_population.append(population[i])
            population = np.array(new_population)

        best_idx = np.argmin(fitness)
        return population[best_idx]