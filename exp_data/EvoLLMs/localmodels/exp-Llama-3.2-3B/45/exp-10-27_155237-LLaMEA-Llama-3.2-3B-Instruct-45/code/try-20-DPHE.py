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
            refined_solution = self.refine_solution(res.x, self.budget)
            return refined_solution
        else:
            return None

    def refine_solution(self, solution, budget):
        # Calculate the probability of refinement
        probability = 0.45

        # Initialize the refined solution
        refined_solution = solution

        # Refine the solution with probabilistic refinement
        for _ in range(int(budget * probability)):
            # Generate a random index
            idx = np.random.randint(0, self.dim)

            # Generate a random perturbation
            perturbation = np.random.uniform(-1.0, 1.0)

            # Apply the perturbation to the refined solution
            refined_solution[idx] += perturbation

            # Ensure the solution remains within the bounds
            refined_solution[idx] = np.clip(refined_solution[idx], self.lower_bound, self.upper_bound)

        return refined_solution

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