import numpy as np
import random
import copy
from collections import deque

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

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
        
        # Initialize the solution
        sol = None
        
        # Try different initializations
        for _ in range(10):
            # Randomly initialize the solution
            sol = np.random.uniform(bounds, size=self.dim)
            
            # Evaluate the function at the solution
            func_sol = self.__call__(func, sol)
            
            # Check if the solution is better than the current best
            if func_sol < self.__call__(func, sol):
                # Update the solution
                sol = sol
        
        # Return the best solution found
        return sol

def mutate(individual):
    # Randomly mutate the individual
    mutated_individual = individual.copy()
    for i in range(len(individual)):
        if random.random() < 0.25:
            mutated_individual[i] = random.uniform(-5.0, 5.0)
    return mutated_individual

def crossover(parent1, parent2):
    # Perform crossover between two parents
    child = parent1.copy()
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child[i] = parent2[i]
    return child

def selection(population):
    # Select the fittest individuals
    fittest_individuals = sorted(population, key=lambda individual: self.evaluate_fitness(individual), reverse=True)
    return fittest_individuals[:int(len(population) * 0.75)]

def mutation_exp(individual, mutation_rate):
    # Perform mutation on the individual
    mutated_individual = individual.copy()
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            mutated_individual[i] = random.uniform(-5.0, 5.0)
    return mutated_individual

class BBOBMetaheuristicEvolutionaryAlgorithm(BBOBMetaheuristic):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.population_size = 100
        self.mutation_rate = 0.1
        self.selection_rate = 0.5

    def __call__(self, func):
        # Initialize the population
        population = [copy.deepcopy(func) for _ in range(self.population_size)]
        
        # Perform selection, crossover, and mutation
        for _ in range(100):
            # Select the fittest individuals
            fittest_individuals = selection(population)
            
            # Create new individuals
            new_individuals = []
            for _ in range(self.population_size):
                parent1, parent2 = random.sample(fittest_individuals, 2)
                child = crossover(parent1, parent2)
                new_individual = mutation_exp(child, self.mutation_rate)
                new_individuals.append(new_individual)
            
            # Mutate the new individuals
            new_individuals = [mutation_exp(individual, self.mutation_rate) for individual in new_individuals]
            
            # Replace the old population with the new population
            population = new_individuals
        
        # Evaluate the function at each individual
        for individual in population:
            func_evals = self.func_evals
            func_evals += 1
            individual = self.__call__(func, individual)
            self.func_evals = func_evals
        
        # Return the best solution found
        return population[0]

# Example usage
if __name__ == "__main__":
    algorithm = BBOBMetaheuristicEvolutionaryAlgorithm(100, 10)
    func = lambda x: x**2
    print(algorithm.search(func))