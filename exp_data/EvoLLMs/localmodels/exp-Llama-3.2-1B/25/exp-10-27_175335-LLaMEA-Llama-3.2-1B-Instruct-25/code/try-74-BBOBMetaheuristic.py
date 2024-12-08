import numpy as np
import random
from scipy.optimize import differential_evolution

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

    def mutate(self, individual):
        # Randomly mutate the individual
        mutation_rate = 0.1
        mutated_individual = individual.copy()
        
        # Randomly select a mutation point
        mutation_point = random.randint(0, self.dim - 1)
        
        # Swap the mutation point with a random point in the search space
        mutated_individual[mutation_point], mutated_individual[mutation_point + random.randint(-10, 10)] = mutated_individual[mutation_point + random.randint(-10, 10)], mutated_individual[mutation_point]
        
        # Apply the mutation
        mutated_individual = np.clip(mutated_individual, bounds, None)
        
        # Return the mutated individual
        return mutated_individual

    def evolve(self, num_generations):
        # Initialize the population
        population = [self.search(func) for _ in range(100)]
        
        # Evolve the population
        for _ in range(num_generations):
            # Calculate the fitness of each individual
            fitnesses = [individual.f for individual in population]
            
            # Select the fittest individuals
            fittest_indices = np.argsort(fitnesses)[-10:]
            fittest_individuals = [population[i] for i in fittest_indices]
            
            # Create new offspring
            offspring = []
            while len(offspring) < 20:
                # Randomly select two parents
                parent1, parent2 = random.sample(fittest_individuals, 2)
                
                # Mutate the parents
                mutated_parent1 = self.mutate(parent1)
                mutated_parent2 = self.mutate(parent2)
                
                # Create the offspring
                offspring.append(mutated_parent1 + mutated_parent2)
            
            # Replace the least fit individuals with the new offspring
            population = [individual for individual in population if individual.f < fitnesses[-1]] + offspring
        
        # Return the best individual
        return population[0]

# One-line description with the main idea
# BBOBMetaheuristic: An evolutionary algorithm for black box optimization using genetic programming.
# The algorithm evolves a population of individuals, each of which is a solution to the optimization problem.
# The individuals are mutated and evaluated to improve their fitness.
# The algorithm selects the fittest individuals and creates new offspring to replace the least fit individuals.
# The process is repeated for a specified number of generations.