import numpy as np

class DynamicMutationGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.base_mutation_rate = 0.1
        self.crossover_rate = 0.9

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget):
            fitness_sorted_idx = np.argsort(fitness)
            fittest_individual = population[fitness_sorted_idx[0]]
            
            new_population = [fittest_individual]
            for _ in range(1, self.population_size):
                parent1 = population[np.random.choice(self.population_size)]
                parent2 = population[np.random.choice(self.population_size)]
                parent3 = population[np.random.choice(self.population_size)]
                mutation_rate = self.base_mutation_rate / (1 + np.sum(fitness) - fitness_sorted_idx)
                child = parent1 + mutation_rate * (parent2 - parent3)
                if np.random.uniform() < mutation_rate:
                    child += np.random.normal(0, 1, self.dim)
                if np.random.uniform() < self.crossover_rate:
                    cross_points = np.random.randint(0, 2, self.dim).astype(bool)
                    child[cross_points] = parent1[cross_points]
                new_population.append(child)
            
            population = np.array(new_population)
            fitness = np.array([func(individual) for individual in population])
        
        best_idx = np.argmin(fitness)
        return population[best_idx]