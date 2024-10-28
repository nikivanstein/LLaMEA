import numpy as np

class AdaptiveGeneticAlgorithm:
    def __init__(self, budget, dim, mutation_rate=0.1, crossover_rate=0.5):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.population_size = 100
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = None
        self.fitness_scores = []

    def __call__(self, func, logger=None):
        if logger is None:
            logger = aoc_logger(self.budget, upper=1e2, triggers=['ALWAYS'])
        
        # Evaluate the function with the given budget
        func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], self.population_size)])
        
        # Select the top-performing individuals
        top_individuals = np.argsort(func_evaluations)[-self.population_size:]
        
        # Create a new population by crossover and mutation
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(top_individuals, 2)
            child = (parent1 + parent2) / 2
            if random.random() < self.mutation_rate:
                child = random.uniform(self.search_space[0], self.search_space[1])
            new_population.append(child)
        
        # Replace the old population with the new one
        self.population = new_population
        
        # Evaluate the new population
        new_func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], len(new_population))])
        
        # Update the fitness scores
        self.fitness_scores = np.array([func_evaluations] + new_func_evaluations)
        
        # Return the best individual
        best_individual = np.argmax(self.fitness_scores)
        return new_population[best_individual]

def aoc_logger(budget, upper=1e2, triggers=['ALWAYS']):
    #... (your existing code)

def bbob_optimization(budget, dim, func):
    algorithm = AdaptiveGeneticAlgorithm(budget, dim)
    return algorithm.__call__(func)

# Example usage
func = lambda x: x**2
print(bbob_optimization(100, 10, func))