import numpy as np

class PEAHM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.hypermutation_rate = 0.3

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        
        # Evaluate initial population
        fitness = np.array([func(x) for x in population])
        
        # Main loop
        for _ in range(self.budget):
            # Select parents
            parents = np.random.choice(population, size=self.population_size, p=fitness/np.sum(fitness))
            
            # Crossover
            offspring = []
            for _ in range(self.population_size):
                parent1, parent2 = np.random.choice(parents, size=2, replace=False)
                child = np.mean([parent1, parent2], axis=0)
                if np.random.rand() < self.crossover_rate:
                    child += np.random.uniform(-5.0, 5.0, self.dim)
                offspring.append(child)
            
            # Hypermutation
            for i in np.random.choice(offspring, size=int(self.population_size*self.hypermutation_rate), replace=False):
                if np.random.rand() < self.hypermutation_rate:
                    i += np.random.uniform(-5.0, 5.0, self.dim)
                offspring[i] = np.clip(offspring[i], -5.0, 5.0)
            
            # Evaluate offspring
            fitness_offspring = np.array([func(x) for x in offspring])
            
            # Replace worst individual
            idx = np.argmin(fitness_offspring)
            population[idx] = offspring[idx]
            
            # Update fitness
            fitness = np.array([func(x) for x in population])
        
        # Return best individual
        idx = np.argmin(fitness)
        return population[idx]

# Example usage:
def func(x):
    return np.sum(x**2)

pea_hm = PEAHM(budget=10, dim=5)
x_opt = pea_hm(func)
print(x_opt)