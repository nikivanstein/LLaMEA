import numpy as np
import random
import os
from scipy.stats import randint

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

class EvolutionaryBBOBMetaheuristic(BBOBMetaheuristic):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.population_size = 100
        self.mutation_rate = 0.01
        self.population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.population_size)]

    def __call__(self, func):
        # Evaluate the function within the budget
        func_evals = self.func_evals
        self.func_evals += 1
        return func

    def mutate(self, individual):
        # Randomly select an index to mutate
        idx = random.randint(0, self.population_size - 1)
        
        # Randomly choose a mutation point
        mutation_point = random.randint(0, self.dim - 1)
        
        # Swap the values at the mutation point
        self.population[idx] = np.append(self.population[idx], self.population[idx][mutation_point])
        
        # Update the fitness of the individual
        self.population[idx][mutation_point] = self.__call__(func, self.population[idx])[mutation_point]

    def evaluate_fitness(self, individual):
        # Evaluate the function at the individual
        func_evals = self.func_evals
        self.func_evals += 1
        return self.__call__(func, individual)

    def select_parents(self):
        # Select parents using tournament selection
        parents = []
        for _ in range(self.population_size // 2):
            # Select a random individual
            individual = random.choice(self.population)
            
            # Select a random individual
            second_individual = random.choice(self.population)
            
            # Calculate the fitness of the two individuals
            fitness_individual = self.evaluate_fitness(individual)
            fitness_second_individual = self.evaluate_fitness(second_individual)
            
            # Select the individual with the higher fitness
            if fitness_individual > fitness_second_individual:
                parents.append(individual)
            else:
                parents.append(second_individual)
        
        # Return the parents
        return parents

    def crossover(self, parents):
        # Perform crossover using roulette wheel selection
        offspring = []
        for _ in range(self.population_size):
            # Select a random individual from the parents
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            
            # Calculate the fitness of the two parents
            fitness1 = self.evaluate_fitness(parent1)
            fitness2 = self.evaluate_fitness(parent2)
            
            # Select the individual with the higher fitness
            if fitness1 > fitness2:
                offspring.append(parent1)
            else:
                offspring.append(parent2)
        
        # Return the offspring
        return offspring

    def mutate_crossover(self, offspring):
        # Perform mutation using point mutation
        for i in range(self.population_size):
            # Randomly select an individual
            individual = offspring[i]
            
            # Randomly choose a mutation point
            mutation_point = random.randint(0, self.dim - 1)
            
            # Swap the values at the mutation point
            individual[mutation_point] = self.__call__(func, individual)[mutation_point]
        
        # Return the offspring
        return offspring

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Code: 
# ```python
def roulette_wheel_selection(population):
    # Calculate the fitness of each individual
    fitnesses = [self.evaluate_fitness(individual) for individual in population]
    
    # Select the individual with the highest fitness
    selection = random.choices(population, weights=fitnesses, k=1)[0]
    
    # Return the selected individual
    return selection

def pointMutation(individual, mutation_rate):
    # Randomly select an index to mutate
    idx = random.randint(0, self.dim - 1)
    
    # Randomly choose a mutation point
    mutation_point = random.randint(0, self.dim - 1)
    
    # Swap the values at the mutation point
    individual[idx] = self.__call__(func, individual)[mutation_point]
    
    # Return the mutated individual
    return individual

def crossover(parent1, parent2):
    # Perform crossover using roulette wheel selection
    selection = roulette_wheel_selection([parent1, parent2])
    
    # Select the parent with the higher fitness
    if self.evaluate_fitness(selection) > self.evaluate_fitness(parent1):
        return parent1
    else:
        return parent2

def mutate_crossover(offspring):
    # Perform mutation using point mutation
    for individual in offspring:
        individual = pointMutation(individual, self.mutation_rate)
    
    # Return the mutated offspring
    return offspring

# Initialize the evolutionary algorithm
evolutionary_bbobmetaheuristic = EvolutionaryBBOBMetaheuristic(100, 10)

# Run the evolutionary algorithm
for _ in range(1000):
    # Select the parents
    parents = evolutionary_bbobmetaheuristic.select_parents()
    
    # Perform crossover
    offspring = evolutionary_bbobmetaheuristic.crossover(parents)
    
    # Perform mutation
    offspring = evolutionary_bbobmetaheuristic.mutate_crossover(offspring)
    
    # Evaluate the fitness of the offspring
    fitnesses = [self.evaluate_fitness(individual) for individual in offspring]
    
    # Select the best individual
    selection = roulette_wheel_selection(offspring)
    
    # Update the evolutionary algorithm
    evolutionary_bbobmetaheuristic.population = selection

# Print the final best individual
best_individual = evolutionary_bbobmetaheuristic.population[0]
print("Best individual:", best_individual)
print("Best fitness:", self.evaluate_fitness(best_individual))