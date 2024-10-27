import numpy as np
import random
import math
from scipy.optimize import differential_evolution

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.population_size = 50
        self.mutation_rate = 0.01

    def __call__(self, func):
        # Check if the function can be evaluated within the budget
        if self.func_evals >= self.budget:
            raise ValueError("Not enough evaluations left to optimize the function")

        # Evaluate the function within the budget
        func_evals = self.func_evals
        self.func_evals += 1
        return func

    def search(self, func):
        # Define the search space
        bounds = np.linspace(-5.0, 5.0, self.dim, endpoint=False)
        
        # Initialize the population
        population = [random.uniform(bounds, size=self.dim) for _ in range(self.population_size)]
        
        # Evolve the population
        for _ in range(100):
            # Calculate the fitness of each individual
            fitness = [self.__call__(func, individual) for individual in population]
            
            # Select the fittest individuals
            fittest = [individual for individual, fitness in zip(population, fitness) if fitness == max(fitness)]
            
            # Create a new generation
            new_population = []
            while len(new_population) < self.population_size:
                # Select two parents using tournament selection
                parent1, parent2 = random.sample(fittest, 2)
                
                # Calculate the fitness of the parents
                fitness1, fitness2 = self.__call__(func, parent1), self.__call__(func, parent2)
                
                # Select the best parent using probability 0.25
                if random.random() < 0.25:
                    new_population.append(parent1)
                else:
                    new_population.append(parent2)
            
            # Crossover (reproduce) the new population
            new_population = [np.array(individual) for individual in new_population]
            
            # Mutate the new population
            for individual in new_population:
                if random.random() < self.mutation_rate:
                    individual[random.randint(0, self.dim-1)] = random.uniform(-5.0, 5.0)
            
            # Replace the old population with the new population
            population = new_population
        
        # Return the best solution found
        return self.__call__(func, population[0])

# One-line description with the main idea
# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with Evolutionary Metaheuristics
# Code: 