import numpy as np

class EnhancedMutationGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.elitism_ratio = 0.2
        self.mutation_prob = 0.1
        self.initial_mutation_rate = 0.1
        self.mutation_decay = 0.95

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
                child = parent1 + self.initial_mutation_rate * (parent2 - parent1)
                if np.random.uniform() < self.mutation_prob:
                    child += np.random.normal(0, 1, self.dim)
                new_population.append(child)
            
            population = np.array(new_population)
            fitness = np.array([func(individual) for individual in population])
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]
            
            # Elitism: Preserve the best individuals
            num_elites = int(self.elitism_ratio * self.population_size)
            elite_indices = np.argsort(fitness)[:num_elites]
            population[:num_elites] = population[elite_indices]
            
            # Dynamic mutation rate based on fitness
            average_fitness = np.mean(fitness)
            self.initial_mutation_rate *= np.exp((average_fitness - fitness[best_idx]) * self.mutation_decay)
        
        return best_individual