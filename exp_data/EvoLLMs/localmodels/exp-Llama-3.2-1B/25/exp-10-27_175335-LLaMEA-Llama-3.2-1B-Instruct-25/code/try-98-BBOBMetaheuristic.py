import numpy as np
import random
import copy

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
            
            # Refine the solution using probability 0.25
            for _ in range(10):
                # Randomly perturb the current solution
                perturbation = random.uniform(-1.0, 1.0)
                
                # Apply the perturbation and check if the solution is better
                new_sol = sol + perturbation
                
                # Evaluate the function at the new solution
                func_sol = self.__call__(func, new_sol)
                
                # Check if the new solution is better than the current best
                if func_sol < self.__call__(func, sol):
                    # Update the solution
                    sol = new_sol
        
        # Return the best solution found
        return sol

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Code: 
# ```python
def fitness(individual, func, budget):
    return self.__call__(func, individual)

def mutation(individual, func, budget):
    perturbation = random.uniform(-1.0, 1.0)
    new_individual = copy.deepcopy(individual)
    for i in range(len(individual)):
        if random.random() < 0.25:
            new_individual[i] += perturbation
    return new_individual

def selection(population, func, budget):
    fitnesses = [fitness(individual, func, budget) for individual in population]
    selected_indices = np.argsort(fitnesses)[-budget:]
    selected_population = [individual for index in selected_indices for individual in population[index]]
    return selected_population

def crossover(parent1, parent2, budget):
    child = copy.deepcopy(parent1)
    for _ in range(10):
        # Randomly select a gene to crossover
        gene_index = random.randint(0, len(parent1) - 1)
        
        # Crossover the genes
        child[gene_index] = parent2[gene_index]
    return child

def evolve(population, func, budget):
    while self.func_evals < budget:
        # Select a parent
        parent1 = random.choice(population)
        parent2 = random.choice(population)
        
        # Crossover the parents
        child = crossover(parent1, parent2, budget)
        
        # Mutate the child
        child = mutation(child, func, budget)
        
        # Add the child to the population
        population.append(child)
    
    # Return the best solution found
    return population[-1]

# Example usage:
budget = 100
dim = 10
func = lambda x: x**2
population = []
for _ in range(100):
    individual = random.uniform(-5.0, 5.0, size=dim)
    population.append(individual)

best_individual = evolve(population, func, budget)
print("Best individual:", best_individual)
print("Fitness:", fitness(best_individual, func, budget))