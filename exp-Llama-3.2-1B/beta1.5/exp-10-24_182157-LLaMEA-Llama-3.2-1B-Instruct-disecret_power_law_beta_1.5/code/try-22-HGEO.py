import numpy as np
import random

class HGEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.budgets = np.arange(budget + 1)
        self.grid_size = 10  # hypergrid size
        self.grid = np.zeros((dim, self.grid_size, self.grid_size))
        self.hypergrids = []
        self.population = []
        self.fitness_history = []

    def __call__(self, func):
        def f(x):
            return np.array(func(x))
        
        # Initialize the population with random individuals
        self.population = [[np.random.rand(self.dim) for _ in range(self.dim)] for _ in range(self.budget)]
        
        # Evaluate the function at each hypergrid location
        for i in range(self.budget):
            x = np.random.rand(self.dim)
            self.grid[:, :, i] = x
            
            # Generate new hypergrids by perturbing the current location
            for j in range(self.grid_size):
                for k in range(self.grid_size):
                    self.grid[:, :, i + 1] = self.grid[:, :, i] + np.random.uniform(-1, 1, self.dim)
            
            # Evaluate the function at each new location
            y = np.array([f(x) for x in np.random.rand(self.dim)])
            
            # Update the hypergrid and its budget
            self.grid[:, :, i + 1] = x
            self.budgets[i + 1] = min(self.budgets[i] + 1, self.budget)
            
            # Add the new hypergrid to the list of hypergrids
            self.hypergrids.append(self.grid[:, :, i + 1])
        
        # Evaluate the function at the last hypergrid location
        x = np.random.rand(self.dim)
        self.grid[:, :, -1] = x
        y = np.array([f(x) for x in np.random.rand(self.dim)])
        
        # Return the optimal solution
        return np.argmax(y)

    def mutate(self, individual):
        mutated_individual = individual.copy()
        for _ in range(self.dim):
            if random.random() < 0.1:  # 10% chance of mutation
                mutated_individual[_] += np.random.uniform(-1, 1, self.dim)
        return mutated_individual

    def crossover(self, parent1, parent2):
        child = parent1.copy()
        for _ in range(self.dim):
            if random.random() < 0.5:  # 50% chance of crossover
                child[_] = parent2[_]
        return child

    def select(self, population):
        fitnesses = [np.argmax(y) for y in population]
        return np.random.choice(self.budget, size=len(fitnesses), p=fitnesses)

    def optimize(self, func):
        while True:
            # Select the fittest individuals
            fittest_individuals = self.select(self.population)
            
            # Evaluate the fitness of each individual
            fitnesses = [np.argmax(y) for y in fittest_individuals]
            
            # Create a new population by mutating and crossing over the fittest individuals
            new_population = []
            for _ in range(self.budget):
                parent1, parent2 = random.sample(fittest_individuals, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            # Replace the old population with the new one
            self.population = new_population
            
            # Evaluate the new population
            fitnesses = [np.argmax(y) for y in self.population]
            
            # If the new population is better, stop
            if np.max(fitnesses) > 0.95 * np.min(fitnesses):
                break
        
        # Return the optimal solution
        return np.argmax(self.population[-1])

# Example usage:
if __name__ == "__main__":
    # Define a black box function
    def func(x):
        return x[0]**2 + x[1]**2
    
    # Create an HGEO instance with 10 budget evaluations and 3 dimensions
    hgeo = HGEO(10, 3)
    
    # Optimize the function using HGEO
    optimal_x = hgeo(func)
    print("Optimal solution:", optimal_x)

# One-line description: Novel metaheuristic algorithm for black box optimization using hypergrid search and mutation/crossover strategies