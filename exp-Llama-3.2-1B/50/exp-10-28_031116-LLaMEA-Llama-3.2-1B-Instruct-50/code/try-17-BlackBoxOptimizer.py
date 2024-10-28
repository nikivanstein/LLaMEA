import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []

    def __call__(self, func):
        # Initialize the population with random solutions
        for _ in range(self.budget):
            # Generate a random solution within the search space
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            # Evaluate the function at the current solution
            func(solution)

        # Select the best solution based on the budget
        best_solution = np.argmax(func)

        # Refine the solution using the best solution as a reference
        for _ in range(self.budget):
            # Get the current best solution
            best_solution_ref = np.argmax(func)
            # Generate a new solution by perturbing the best solution
            perturbed_solution = best_solution_ref + np.random.uniform(-1.0, 1.0, self.dim)
            # Evaluate the function at the perturbed solution
            func(perturbed_solution)

            # If the perturbed solution is better, replace the current best solution
            if func[perturbed_solution] > func[best_solution]:
                best_solution = perturbed_solution

        # Update the population with the best solution
        self.population.append(best_solution)

        # Return the best solution found so far
        return best_solution

    def select_solution(self, num_solutions):
        # Select the top num_solutions solutions from the population
        selected_solutions = np.random.choice(self.population, num_solutions, replace=False)
        # Return the top num_solutions solutions
        return selected_solutions