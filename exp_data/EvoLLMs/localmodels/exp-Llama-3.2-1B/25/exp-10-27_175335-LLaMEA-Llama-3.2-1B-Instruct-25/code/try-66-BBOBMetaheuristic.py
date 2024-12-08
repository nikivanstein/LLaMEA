import numpy as np

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.population = None

    def __call__(self, func):
        # Check if the function can be evaluated within the budget
        if self.func_evals >= self.budget:
            raise ValueError("Not enough evaluations left to optimize the function")

        # Evaluate the function within the budget
        func_evals = self.func_evals
        self.func_evals += 1
        return func

    def search(self, func, num_individuals):
        # Define the search space
        bounds = np.linspace(-5.0, 5.0, self.dim, endpoint=False)
        
        # Initialize the population
        self.population = [np.random.uniform(bounds, size=self.dim) for _ in range(num_individuals)]
        
        # Run the evolution process
        for _ in range(1000):
            # Evaluate the function at the population
            func_population = [self.__call__(func, individual) for individual in self.population]
            
            # Select the fittest individuals
            self.population = [individual for _, individual in sorted(zip(func_population, self.population), reverse=True)[:self.budget]]
            
            # Mutate the population
            for individual in self.population:
                if np.random.rand() < 0.25:
                    self.population[self.population.index(individual)] = np.random.uniform(bounds, size=self.dim)
        
        # Return the best solution found
        return self.population[0]

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Code: 