import numpy as np

class AcceleratedMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = [func(ind) for ind in population]
        
        for _ in range(self.budget):
            sorted_indices = np.argsort(fitness)
            fittest = population[sorted_indices[0]]
            pop_mean = np.mean(population, axis=0)
            diversity = np.mean(np.linalg.norm(population - pop_mean, axis=1))

            mutation_strength = 5.0 / (1.0 + diversity)
            crossover_prob = 0.8

            mutated = population + mutation_strength * np.random.randn(self.budget, self.dim)
            crossover_mask = np.random.choice([0, 1], size=(self.budget, self.dim), p=[1 - crossover_prob, crossover_prob])
            crossover_indices = np.random.choice(self.budget, size=int(self.budget * 0.1), replace=False)

            crossover_population = population[crossover_indices] + crossover_mask[crossover_indices] * (mutated[crossover_indices] - population[crossover_indices])
            crossover_fitness = [func(ind) for ind in crossover_population]
            
            for i in range(self.budget):
                if crossover_fitness[i] < fitness[i]:
                    population[i] = crossover_population[i]
                    fitness[i] = crossover_fitness[i]

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        return best_solution