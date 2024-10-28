import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(-5.0, 5.0, self.dim)
            # Evaluate the function at the point
            value = func(point)
            # Check if the point is within the bounds
            if -5.0 <= point[0] <= 5.0 and -5.0 <= point[1] <= 5.0:
                # If the point is within bounds, update the function value
                self.func_evals += 1
                return value
        # If the budget is exceeded, return the best point found so far
        return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

    def iterated_permutation(self, func, budget):
        # Initialize the population with random points in the search space
        population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(budget)]
        
        # Repeat the process until the budget is exceeded
        while self.func_evals < budget:
            # Initialize the new population with the best individual from the previous generation
            new_population = population[:self.dim]
            
            # Generate a new population by iterated permutation
            for _ in range(self.dim):
                # Select two random parents from the new population
                parent1, parent2 = np.random.choice(population, 2, replace=False)
                
                # Create a new child by crossover
                child = parent1[:self.dim//2] + parent2[self.dim//2:]
                
                # Create a new individual by mutation
                mutated_child = np.random.uniform(-5.0, 5.0, self.dim)
                mutated_child[self.dim//2:] = child
                
                # Evaluate the new individual
                value = func(mutated_child)
                
                # Check if the new individual is within the bounds
                if -5.0 <= mutated_child[0] <= 5.0 and -5.0 <= mutated_child[1] <= 5.0:
                    # If the new individual is within bounds, update the new population
                    new_population.append(mutated_child)
            
            # Update the population with the new population
            population = new_population
        
        # Return the best individual from the final population
        return np.max(population)

    def iterated_cooling(self, func, budget):
        # Initialize the population with random points in the search space
        population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(budget)]
        
        # Repeat the process until the budget is exceeded
        while self.func_evals < budget:
            # Initialize the new population with the best individual from the previous generation
            new_population = population[:self.dim]
            
            # Generate a new population by iterated cooling
            for _ in range(self.dim):
                # Select two random parents from the new population
                parent1, parent2 = np.random.choice(population, 2, replace=False)
                
                # Create a new child by crossover
                child = parent1[:self.dim//2] + parent2[self.dim//2:]
                
                # Create a new individual by mutation
                mutated_child = np.random.uniform(-5.0, 5.0, self.dim)
                mutated_child[self.dim//2:] = child
                
                # Evaluate the new individual
                value = func(mutated_child)
                
                # Check if the new individual is within the bounds
                if -5.0 <= mutated_child[0] <= 5.0 and -5.0 <= mutated_child[1] <= 5.0:
                    # If the new individual is within bounds, update the new population
                    new_population.append(mutated_child)
            
            # Update the population with the new population
            population = new_population
            
            # Apply cooling to the new population
            self.func_evals = min(self.func_evals + 1, budget)
        
        # Return the best individual from the final population
        return np.max(population)