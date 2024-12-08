import numpy as np

class EnhancedMetaheuristicAlgorithm:
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

            mutated = population + mutation_strength * np.random.randn(self.budget, self.dim)
            mutated_fitness = [func(ind) for ind in mutated]

            performance = np.array(fitness) / np.array(mutated_fitness)
            selection_prob = 1 - (performance - np.min(performance)) / (np.max(performance) - np.min(performance))

            selected_indices = np.random.choice(range(self.budget), size=self.budget, p=selection_prob)
            for i in range(self.budget):
                if mutated_fitness[selected_indices[i]] < fitness[i]:
                    population[i] = mutated[selected_indices[i]]
                    fitness[i] = mutated_fitness[selected_indices[i]]

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        return best_solution