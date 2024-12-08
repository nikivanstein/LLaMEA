import numpy as np

class EnhancedAdaptiveMutationGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.mutation_prob = 0.1
        self.crossover_rate = 0.9

    def _local_search(self, population, func):
        for i in range(self.population_size):
            candidate = population[i].copy()
            for _ in range(3):
                new_candidate = candidate + 0.1 * np.random.normal(0, 1, self.dim)
                if func(new_candidate) < func(candidate):
                    candidate = new_candidate
            population[i] = candidate

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        mutation_rate = np.full(self.population_size, 0.1)
        
        for _ in range(self.budget):
            fitness = np.array([func(individual) for individual in population])
            fitness_sorted_idx = np.argsort(fitness)
            fittest_individual = population[fitness_sorted_idx[0]]
            
            new_population = [fittest_individual]
            for i in range(1, self.population_size):
                parents_idx = np.random.choice(self.population_size, 3, replace=False)
                parent1, parent2, parent3 = population[parents_idx]
                child = parent1 + mutation_rate[i] * (parent2 - parent3)
                if np.random.uniform() < self.mutation_prob:
                    child += np.random.normal(0, 1, self.dim)
                if np.random.uniform() < self.crossover_rate:
                    cross_points = np.random.randint(0, 2, self.dim).astype(bool)
                    child[cross_points] = parent1[cross_points]
                new_population.append(child)
                
                # Update mutation rate for each individual
                if i > 1:
                    mutation_rate[i] = np.clip(mutation_rate[i] + np.random.normal(0, 0.1), 0.01, 0.5)
            
            population = np.array(new_population)

            self._local_search(population, func)
        
        best_idx = np.argmin([func(individual) for individual in population])
        return population[best_idx]