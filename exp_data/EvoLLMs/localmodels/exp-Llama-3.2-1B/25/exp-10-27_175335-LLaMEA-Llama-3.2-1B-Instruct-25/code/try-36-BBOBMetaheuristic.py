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

    def mutate(self, func):
        # Randomly mutate the solution
        bounds = np.linspace(-5.0, 5.0, self.dim, endpoint=False)
        sol = np.random.uniform(bounds, size=self.dim)
        sol = np.clip(sol, -5.0, 5.0)  # Clip to bounds
        func_sol = self.__call__(func, sol)
        return func_sol

    def crossover(self, parent1, parent2):
        # Perform crossover
        bounds = np.linspace(-5.0, 5.0, self.dim, endpoint=False)
        sol1 = np.random.uniform(bounds, size=self.dim)
        sol2 = np.random.uniform(bounds, size=self.dim)
        sol = np.clip(np.random.choice([sol1, sol2], size=self.dim), -5.0, 5.0)
        return self.__call__(func, sol)

    def __call__(self, func, individual):
        # Perform the evolutionary algorithm
        best_individual = individual
        best_fitness = func(best_individual, individual)
        
        for _ in range(self.budget):
            # Try different mutations
            new_individual = self.mutate(func(individual, self))
            new_fitness = func(new_individual, individual)
            
            # Check if the new solution is better than the current best
            if new_fitness > best_fitness:
                # Update the best solution
                best_individual = new_individual
                best_fitness = new_fitness
        
        # Return the best solution found
        return best_individual

# One-line description with main idea
# Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Refines the strategy by changing individual lines of the selected solution to refine its strategy