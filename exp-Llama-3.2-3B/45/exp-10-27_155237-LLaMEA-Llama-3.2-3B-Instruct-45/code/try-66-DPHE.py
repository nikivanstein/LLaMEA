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

    # Refine the strategy using probability-based mutation
    def refine_strategy(individual, logger, mutation_prob=0.45):
        if np.random.rand() < mutation_prob:
            # Select two random individuals
            i, j = np.random.randint(0, len(individual), size=2)
            # Calculate the probability of mutation for each dimension
            mutation_probabilities = np.random.rand(len(individual))
            # Select dimensions to mutate
            mutate_dimensions = np.where(mutation_probabilities > np.random.rand(len(individual)))[0]
            # Mutate the selected dimensions
            individual[mutate_dimensions] = np.random.uniform(individual[mutate_dimensions] - 1, individual[mutate_dimensions] + 1)
        return individual

    # Update the strategy
    selected_algorithm = "DPHE"
    current_population = [("DPHE", "Novel 'Differential Perturbation and Hybrid Evolution' (DPHE) algorithm", -np.inf)]
    selected_algorithm = max(current_population, key=lambda x: x[2])
    new_individual = selected_algorithm[1]
    refined_individual = refine_strategy(new_individual, logger=None, mutation_prob=0.45)
    current_population.append((selected_algorithm[0], refined_individual, -np.inf))
    selected_algorithm = max(current_population, key=lambda x: x[2])