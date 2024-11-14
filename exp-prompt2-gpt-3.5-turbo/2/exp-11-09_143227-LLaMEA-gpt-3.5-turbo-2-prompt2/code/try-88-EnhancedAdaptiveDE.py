import numpy as np

class EnhancedAdaptiveDE(AdaptiveDE):
    def __init__(self, budget, dim, F=0.8, CR=0.9, pop_size=20, diversity_ratio=0.5):
        super().__init__(budget, dim, F, CR, pop_size)
        self.diversity_ratio = diversity_ratio

    def __call__(self, func):
        def adjust_population(population, fitness, diversity_ratio):
            sorted_indices = np.argsort(fitness)
            sorted_population = population[sorted_indices]
            n_top = max(int(diversity_ratio * len(population)), 1)
            top_population = sorted_population[:n_top]
            return np.concatenate([top_population] * (len(population) // n_top) + [top_population[:len(population) % n_top]])

        population = np.random.uniform(-5, 5, (self.pop_size, self.dim))
        fitness = np.array([func(individual) for individual in population])

        for _ in range(self.budget):
            adapt_F = self.F * (1.0 - _ / self.budget)  # Adapt F over time
            adapt_CR = self.CR + 0.1 * np.sin(0.9 * np.pi * _ / self.budget)  # Adapt CR with sinusoidal function
            population = adjust_population(population, fitness, self.diversity_ratio)
            new_population = []
            for i, target in enumerate(population):
                mutant = mutate(target, population, adapt_F)
                trial = crossover(target, mutant, adapt_CR)
                new_fitness = func(trial)
                if new_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = new_fitness
                new_population.append(population[i])
            population = np.array(new_population)

        best_idx = np.argmin(fitness)
        return population[best_idx]