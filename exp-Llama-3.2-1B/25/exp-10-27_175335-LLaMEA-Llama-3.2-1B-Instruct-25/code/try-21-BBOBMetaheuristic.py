import numpy as np
import random
import matplotlib.pyplot as plt
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

def evolution_optimization(func, bounds, budget, dim, mutation_rate, mutation_prob):
    # Initialize the population
    population = [random.uniform(bounds, size=dim) for _ in range(100)]
    
    # Evaluate the function for each individual in the population
    fitness = []
    for individual in population:
        func_evals = 0
        for _ in range(budget):
            func_evals += 1
            func(individual, individual)
        fitness.append(func_evals)
    
    # Evolve the population
    while len(fitness) < budget:
        # Select the fittest individuals
        fittest_individuals = population[np.argsort(fitness)]
        
        # Mutate the fittest individuals
        mutated_individuals = []
        for individual in fittest_individuals:
            for _ in range(mutation_rate):
                # Change a random value in the individual
                mutated_individual = individual.copy()
                mutated_individual[random.randint(0, dim-1)] += random.uniform(-1, 1)
                
                # Check if the mutation is within the bounds
                if mutated_individual[random.randint(0, dim-1)] < bounds[0]:
                    mutated_individual[random.randint(0, dim-1)] += bounds[0]
                elif mutated_individual[random.randint(0, dim-1)] > bounds[1]:
                    mutated_individual[random.randint(0, dim-1)] -= bounds[1]
        
        # Evaluate the function for the mutated individuals
        fitness = [0] * len(population)
        for individual in mutated_individuals:
            func_evals = 0
            for _ in range(budget):
                func_evals += 1
                func(individual, individual)
            fitness.append(func_evals)
        
        # Replace the fittest individuals with the mutated individuals
        population = fittest_individuals[:len(fittest_individuals)//2] + mutated_individuals[len(fittest_individuals)//2:]
    
    # Return the best solution found
    return population[np.argmax(fitness)]

# Test the evolutionary algorithm
func = lambda x: np.sin(x)
bounds = [-5.0, 5.0]
budget = 100
dim = 10
mutation_rate = 0.01
mutation_prob = 0.5

best_individual = evolution_optimization(func, bounds, budget, dim, mutation_rate, mutation_prob)
print("Best individual:", best_individual)
print("Best fitness:", func(best_individual))

# Plot the fitness landscape
plt.figure(figsize=(10, 6))
plt.plot(np.linspace(-5.0, 5.0, 100), np.sin(np.linspace(-5.0, 5.0, 100)))
plt.scatter([best_individual for _ in range(100)], [func(best_individual) for _ in range(100)], c='r')
plt.xlabel('Individual')
plt.ylabel('Fitness')
plt.title('Fitness Landscape')
plt.show()