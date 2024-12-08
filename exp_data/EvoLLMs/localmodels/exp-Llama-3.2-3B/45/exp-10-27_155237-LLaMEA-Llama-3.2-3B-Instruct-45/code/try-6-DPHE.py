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

# Probability-based mutation refinement
def refine_dphe(dphe, population, aucs):
    # Select the top 10% of individuals with the best aucs
    top_individuals = np.argsort(aucs)[-int(0.1*len(aucs)):]
    top_individuals = population[top_individuals]

    # Refine the strategy of the selected individuals
    refined_population = []
    for individual in top_individuals:
        # Calculate the probability of mutation
        prob = 0.45
        if np.random.rand() < prob:
            # Perform mutation
            mutation = np.random.uniform(-0.1, 0.1, size=len(individual))
            individual += mutation
            # Ensure the individual stays within the bounds
            individual = np.clip(individual, dphe.lower_bound, dphe.upper_bound)
        refined_population.append(individual)

    return refined_population

# Evaluate the refined population
def evaluate_refined_population(dphe, refined_population, func):
    aucs = []
    for individual in refined_population:
        # Evaluate the function using the refined individual
        result = dphe(func, individual)
        aucs.append(result)
    return aucs

# Main function
def main():
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

    # Generate the population of algorithms
    population = []
    for _ in range(100):
        individual = np.random.uniform(dphe.lower_bound, dphe.upper_bound, size=dphe.dim)
        population.append(individual)

    # Refine the strategy of the selected individuals
    refined_population = refine_dphe(dphe, population, evaluate_refined_population(dphe, population, func))

    # Evaluate the refined population
    aucs = evaluate_refined_population(dphe, refined_population, func)

    # Print the aucs
    print("Aucs:", aucs)

if __name__ == "__main__":
    main()