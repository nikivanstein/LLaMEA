import numpy as np

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

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim, mutation_rate):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = mutation_rate
        self.func_evals = 0
        self.population = []

    def __call__(self, func):
        # Initialize the population
        self.population = self.generate_population(func, self.budget, self.dim)

    def generate_population(self, func, budget, dim):
        # Create an initial population of random solutions
        population = [np.random.uniform(-5.0, 5.0, size=self.dim) for _ in range(100)]
        
        # Evolve the population over time
        for _ in range(budget):
            # Select the fittest individuals
            fittest = sorted(self.population, key=self.evaluate_fitness, reverse=True)[:self.population.index(max(self.population))]
            
            # Create new individuals by mutating the fittest solutions
            new_population = [f + self.mutation(f, self.mutation_rate) for f in fittest]
            
            # Add the new individuals to the population
            self.population.extend(new_population)
        
        # Return the final population
        return self.population

    def evaluate_fitness(self, individual):
        # Evaluate the fitness of an individual
        func_evals = 0
        func_sol = self.__call__(func, individual)
        return func_sol

    def mutation(self, individual, mutation_rate):
        # Apply mutation to an individual
        new_individual = individual.copy()
        if np.random.rand() < mutation_rate:
            # Randomly select a gene to mutate
            gene = np.random.choice(self.dim)
            
            # Mutate the gene
            new_individual[gene] += np.random.uniform(-1, 1)
        
        return new_individual

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Code: 