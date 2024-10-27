import numpy as np
import random

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

    def search(self, func, bounds):
        # Define the search space
        self.bounds = bounds
        
        # Initialize the solution
        sol = None
        
        # Try different initializations
        for _ in range(10):
            # Randomly initialize the solution
            sol = np.random.uniform(self.bounds, size=self.dim)
            
            # Evaluate the function at the solution
            func_sol = self.__call__(func, sol)
            
            # Check if the solution is better than the current best
            if func_sol < self.__call__(func, sol):
                # Update the solution
                sol = sol
        
        # Return the best solution found
        return sol

class GeneticBBOBMetaheuristic(BBOBMetaheuristic):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.population_size = 50
        self.mutation_rate = 0.1
        self.population = self.initialize_population()

    def initialize_population(self):
        # Initialize the population with random solutions
        population = [self.search(func, self.bounds) for _ in range(self.population_size)]
        return population

    def fitness(self, individual):
        # Evaluate the function at the individual
        func = self.search(func, self.bounds)
        return func(individual)

    def mutate(self, individual):
        # Randomly mutate the individual
        mutated_individual = individual.copy()
        if random.random() < self.mutation_rate:
            mutated_individual[random.randint(0, self.dim - 1)] = random.uniform(self.bounds[0], self.bounds[1])
        return mutated_individual

    def crossover(self, parent1, parent2):
        # Perform crossover to create a new offspring
        offspring = parent1[:self.dim // 2] + parent2[self.dim // 2:]
        return offspring

    def evaluate_fitness(self, population):
        # Evaluate the fitness of each individual in the population
        fitnesses = [self.fitness(individual) for individual in population]
        return fitnesses

    def select_parents(self, fitnesses):
        # Select parents based on their fitness
        parents = [individual for _, individual in sorted(zip(fitnesses, population), reverse=True)]
        return parents

    def mutate_population(self, population):
        # Mutate the population
        mutated_population = []
        for individual in population:
            mutated_individual = self.mutate(individual)
            mutated_population.append(mutated_individual)
        return mutated_population

    def replace_population(self, mutated_population, population):
        # Replace the old population with the mutated one
        population[:] = mutated_population

    def run(self):
        # Run the genetic algorithm
        fitnesses = self.evaluate_fitness(self.population)
        parents = self.select_parents(fitnesses)
        mutated_population = self.mutate_population(self.population)
        self.replace_population(mutated_population, parents)
        return fitnesses

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Code: 