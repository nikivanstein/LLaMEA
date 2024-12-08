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

    def mutate(self, sol):
        # Select two random points in the search space
        idx1, idx2 = np.random.choice(self.dim, size=2, replace=False)
        
        # Refine the solution by changing one of the points
        if np.random.rand() < 0.25:
            sol[idx1] = np.random.uniform(-5.0, 5.0)
        
        return sol

    def evolve(self, func, population_size):
        # Initialize the population
        population = [self.search(func) for _ in range(population_size)]
        
        # Evolve the population for a fixed number of generations
        for _ in range(100):
            # Select the fittest individuals
            fittest = population[np.argmax([self.func_evals / (i + 1) for i in range(len(population))])]
            
            # Mutate the fittest individuals
            mutated = [self.mutate(individual) for individual in fittest]
            
            # Replace the least fit individuals with the mutated ones
            population = [individual if self.func_evals / (i + 1) < 0.5 else mutated[i] for i, individual in enumerate(population)]
        
        # Return the fittest individual in the final population
        return self.search(func)