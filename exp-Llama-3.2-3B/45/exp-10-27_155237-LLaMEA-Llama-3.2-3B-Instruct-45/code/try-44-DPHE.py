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

def mutation_refinement(individual, logger, mutation_prob):
    if np.random.rand() < mutation_prob:
        # Select two random individuals from the population
        other_individuals = np.random.choice(individuals, size=2, replace=False)
        
        # Calculate the difference between the two individuals
        diff = individual - other_individuals
        
        # Perturb the individual by adding the difference to it
        perturbed_individual = individual + diff
        
        # Replace the individual with the perturbed individual
        individuals = individuals.replace(individual, perturbed_individual)
        
        logger.log(f"Mutation refinement: {individual} -> {perturbed_individual}")
    else:
        logger.log(f"No mutation refinement for {individual}")

def hybrid_evolution(individuals, func, logger, budget):
    # Select the best individual
    best_individual = min(individuals, key=lambda x: func(x))
    
    # Perform differential evolution to refine the best individual
    refined_individual = DPHE(budget=10, dim=individuals.shape[1]).__call__(func)
    
    # If the refined individual is better than the original best individual, replace it
    if func(refined_individual) < func(best_individual):
        individuals = individuals.replace(best_individual, refined_individual)
        
        logger.log(f"Hybrid evolution: {best_individual} -> {refined_individual}")
    else:
        logger.log(f"No hybrid evolution for {best_individual}")

def differential_perturbation(individuals, func, logger, budget):
    # Select the worst individual
    worst_individual = max(individuals, key=lambda x: func(x))
    
    # Perform differential evolution to refine the worst individual
    refined_individual = DPHE(budget=10, dim=individuals.shape[1]).__call__(func)
    
    # If the refined individual is better than the original worst individual, replace it
    if func(refined_individual) < func(worst_individual):
        individuals = individuals.replace(worst_individual, refined_individual)
        
        logger.log(f"Differential perturbation: {worst_individual} -> {refined_individual}")
    else:
        logger.log(f"No differential perturbation for {worst_individual}")

# Example usage:
if __name__ == "__main__":
    # Define a sample black box function
    def func(x):
        return np.sum(x**2)

    # Initialize the DPHE algorithm
    dphe = DPHE(budget=100, dim=10)

    # Initialize the population of individuals
    individuals = np.random.uniform(-5.0, 5.0, size=(100, 10))
    
    # Initialize the logger
    logger = {}
    
    # Perform hybrid evolution
    for _ in range(100):
        hybrid_evolution(individuals, func, logger, dphe.budget)
        
        # Perform mutation refinement
        mutation_refinement(individuals, logger, 0.45)
        
        # Perform differential perturbation
        differential_perturbation(individuals, func, logger, dphe.budget)
        
        # Update the logger
        logger.update({f"auc-{i}": func(individuals) for i in range(len(individuals))})
        
        # Print the best individual
        best_individual = min(individuals, key=lambda x: func(x))
        print(f"Best individual: {best_individual}")