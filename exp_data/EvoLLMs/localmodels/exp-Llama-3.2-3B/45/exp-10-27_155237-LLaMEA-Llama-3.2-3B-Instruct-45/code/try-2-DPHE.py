import numpy as np
from scipy.optimize import differential_evolution

class DPHE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def neg_func(x):
            return -func(x)

        bounds = [(self.lower_bound, self.upper_bound) for _ in range(self.dim)]
        res = differential_evolution(func, bounds, x0=np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim), maxiter=self.budget, tol=1e-6)

        if res.success:
            return res.x
        else:
            return None

# Example usage:
if __name__ == "__main__":
    # Define a sample black box function
    def func(x):
        return np.sum(x**2)

    # Initialize the DPHE algorithm
    dphe = DPHE(budget=100, dim=10)

    # Optimize the function
    result = dphe(func)

    # Print the result
    if result is not None:
        print("Optimal solution:", result)
    else:
        print("Failed to converge")

# Refine the strategy using probability 0.45
def refine_strategy(individual, logger):
    # Select 45% of the individuals to refine their strategy
    indices = np.random.choice(len(individual), size=int(0.45*len(individual)), replace=False)
    refined_individuals = [individual[i] for i in indices]
    
    # Refine the strategy for each selected individual
    for individual in refined_individuals:
        # Perturb the individual's parameters
        perturbed_individual = individual + np.random.uniform(-0.1, 0.1, size=len(individual))
        
        # Evaluate the perturbed individual's fitness
        fitness = func(perturbed_individual)
        
        # If the perturbed individual's fitness is better than the original individual's fitness, update the individual
        if fitness < func(individual):
            individual = perturbed_individual
    
    return individual

# Update the DPHE algorithm's strategy
class DPHERefined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.refine_probability = 0.45

    def __call__(self, func):
        def neg_func(x):
            return -func(x)

        bounds = [(self.lower_bound, self.upper_bound) for _ in range(self.dim)]
        res = differential_evolution(func, bounds, x0=np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim), maxiter=self.budget, tol=1e-6)

        if res.success:
            return res.x
        else:
            return None

        # Refine the strategy for each individual
        refined_individuals = []
        for individual in res.x0:
            refined_individual = refine_strategy(individual, logger)
            refined_individuals.append(refined_individual)
        
        # Evaluate the fitness of each refined individual
        fitnesses = []
        for individual in refined_individuals:
            fitness = func(individual)
            fitnesses.append(fitness)
        
        # Update the best individual based on the fitness values
        best_individual = refined_individuals[np.argmin(fitnesses)]
        
        return best_individual

# Example usage:
if __name__ == "__main__":
    # Define a sample black box function
    def func(x):
        return np.sum(x**2)

    # Initialize the DPHERefined algorithm
    dphe_refined = DPHERefined(budget=100, dim=10)

    # Optimize the function
    result = dphe_refined(func)

    # Print the result
    if result is not None:
        print("Optimal solution:", result)
    else:
        print("Failed to converge")