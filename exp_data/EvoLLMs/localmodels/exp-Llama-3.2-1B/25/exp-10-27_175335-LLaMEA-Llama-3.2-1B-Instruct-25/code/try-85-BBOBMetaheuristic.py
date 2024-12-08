# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Code: 
# ```python
import numpy as np
import random
from copy import deepcopy
from collections import deque

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.population_size = 100
        self.mutation_rate = 0.01
        self.crossover_rate = 0.5

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
        population = self.generate_population(self.population_size, bounds)
        
        # Evolve the population
        for _ in range(100):
            # Select the fittest individuals
            fittest_individuals = sorted(population, key=self.fitness, reverse=True)[:self.population_size // 2]
            
            # Mutate the fittest individuals
            mutated_individuals = []
            for individual in fittest_individuals:
                mutated_individual = self.mutate(individual, bounds)
                mutated_individuals.append(mutated_individual)
            
            # Crossover the mutated individuals
            children = []
            for i in range(0, len(mutated_individuals), 2):
                parent1 = mutated_individuals[i]
                parent2 = mutated_individuals[i + 1]
                child = self.crossover(parent1, parent2)
                children.append(child)
            
            # Replace the least fit individuals with the new ones
            population = self.replace_fittest(population, fittest_individuals, children)
        
        # Return the best solution found
        return self.get_best_solution(population)

    def generate_population(self, size, bounds):
        return [np.random.uniform(bounds, size=size) for _ in range(size)]

    def mutate(self, individual, bounds):
        mutated_individual = individual.copy()
        for _ in range(random.randint(0, self.mutation_rate)):
            idx = random.randint(0, self.dim - 1)
            mutated_individual[idx] += random.uniform(-1, 1)
            if mutated_individual[idx] < -5.0:
                mutated_individual[idx] = -5.0
            elif mutated_individual[idx] > 5.0:
                mutated_individual[idx] = 5.0
        return mutated_individual

    def crossover(self, parent1, parent2):
        # Define the crossover rate
        crossover_rate = self.crossover_rate
        
        # Define the number of offspring
        num_offspring = 2
        
        # Create the offspring
        offspring = []
        for _ in range(num_offspring):
            child = parent1[:self.dim // 2] + parent2[self.dim // 2:]
            if random.random() < crossover_rate:
                # Swap the parents
                idx = random.randint(0, self.dim - 1)
                child[idx], child[self.dim // 2] = parent2[idx], parent1[self.dim // 2]
            offspring.append(child)
        
        return offspring

    def replace_fittest(self, population, fittest_individuals, children):
        # Replace the least fit individuals with the new ones
        population = list(fittest_individuals)
        population.extend(children)
        
        # Sort the population by fitness
        population.sort(key=self.fitness, reverse=True)
        
        # Remove the fittest individuals
        population = population[:self.population_size // 2]
        
        return population

    def get_best_solution(self, population):
        # Return the best solution found
        return population[0]

# One-line description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Code: 
# ```python
# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Code: 
# ```python