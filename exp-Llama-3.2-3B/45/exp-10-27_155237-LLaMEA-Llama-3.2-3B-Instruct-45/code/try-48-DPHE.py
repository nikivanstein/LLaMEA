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
            # Refine the solution with probabilistic refinement
            refined_solution = self.refine_solution(res.x, self.budget, self.dim)
            return refined_solution
        else:
            return None

    def refine_solution(self, solution, budget, dim):
        # Calculate the probability of refinement
        refinement_prob = 0.45

        # Initialize the new solution
        new_solution = solution

        # Refine the solution with probabilistic refinement
        for _ in range(int(budget * refinement_prob)):
            # Generate a new individual by perturbing the solution
            new_individual = self.perturb_solution(solution, dim)

            # Evaluate the fitness of the new individual
            fitness = func(new_individual)

            # If the new individual has a better fitness, update the solution
            if fitness < func(solution):
                new_solution = new_individual

        return new_solution

    def perturb_solution(self, solution, dim):
        # Generate a new individual by perturbing the solution
        new_individual = np.copy(solution)
        for i in range(dim):
            # Randomly perturb the solution
            if np.random.rand() < 0.5:
                new_individual[i] += np.random.uniform(-1.0, 1.0)
            new_individual[i] = np.clip(new_individual[i], self.lower_bound, self.upper_bound)

        return new_individual

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