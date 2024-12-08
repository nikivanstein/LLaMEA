import numpy as np
import random
import copy
import time

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
            sol = copy.deepcopy(bounds)
            for _ in range(self.dim):
                sol = [random.uniform(-5.0, 5.0) for _ in range(self.dim)]
            
            # Evaluate the function at the solution
            func_sol = self.__call__(func, sol)
            
            # Check if the solution is better than the current best
            if func_sol < self.__call__(func, sol):
                # Update the solution
                sol = sol
        
        # Return the best solution found
        return sol

class GeneticProgrammingMetaheuristic(BBOBMetaheuristic):
    def __init__(self, budget, dim, mutation_rate, population_size):
        super().__init__(budget, dim)
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.fitness_scores = []
        self.population = []

    def __call__(self, func):
        # Initialize the population
        for _ in range(self.population_size):
            sol = copy.deepcopy(self.search(func))
            
            # Evaluate the fitness score
            fitness = self.fitness(func, sol)
            
            # Store the fitness score
            self.fitness_scores.append(fitness)
            self.population.append(sol)
        
        # Select the fittest individuals
        self.population = self.select_fittest(self.population, self.fitness_scores)
        
        # Evolve the population
        for _ in range(100):
            # Select parents
            parents = self.select_parents(self.population, self.fitness_scores)
            
            # Crossover
            offspring = self.crossover(parents)
            
            # Mutate the offspring
            offspring = self.mutate(offspring)
            
            # Evaluate the fitness of the offspring
            self.fitness_scores = []
            for sol in offspring:
                fitness = self.fitness(func, sol)
                self.fitness_scores.append(fitness)
                self.population.append(sol)
        
        # Return the best solution found
        return self.population[0]

    def select_fittest(self, population, fitness_scores):
        # Select the fittest individuals
        fittest = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
        
        # Return the fittest individuals
        return [x[0] for x in fittest]

    def select_parents(self, population, fitness_scores):
        # Select parents using tournament selection
        tournament_size = 5
        parents = []
        
        for _ in range(self.population_size):
            winner = random.choice(population)
            tournament = [winner]
            
            for _ in range(tournament_size - 1):
                winner = random.choice(population)
                while winner == tournament[-1]:
                    winner = random.choice(population)
                tournament.append(winner)
            
            parents.append(tournament[-1])
        
        return parents

    def crossover(self, parents):
        # Perform crossover (selection of two parents)
        offspring = []
        
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(parents, 2)
            child = (parent1[:self.dim] + parent2[self.dim:])[:self.dim]
            
            offspring.append(child)
        
        return offspring

    def mutate(self, offspring):
        # Perform mutation (random change in an individual)
        mutated_offspring = []
        
        for sol in offspring:
            mutated_sol = copy.deepcopy(sol)
            
            # Randomly change a single element
            idx = random.randint(0, self.dim - 1)
            mutated_sol[idx] += np.random.uniform(-1, 1)
            
            mutated_offspring.append(mutated_sol)
        
        return mutated_offspring

    def fitness(self, func, sol):
        # Evaluate the fitness of an individual
        return func(sol)

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Code: 