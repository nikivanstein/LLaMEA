import numpy as np
import random

class Noxys:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []
        self.best_func = None
        self.best_score = -np.inf
        self.t = 0

    def __call__(self, func):
        # Initialize population with random solutions
        for _ in range(1000):
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            self.population.append(solution)
        
        # Evaluate population using the given function
        func_evals = [func(x) for x in self.population]
        best_func_idx = np.argmin(func_evals)
        best_func = self.population[best_func_idx]
        best_score = func_evals[best_func_idx]

        # Refine the solution using simulated annealing
        while self.t < self.budget:
            # Generate a new solution by perturbing the current solution
            perturbed_solution = self.population[best_func_idx] + np.random.uniform(-1.0, 1.0, self.dim)
            
            # Evaluate the new solution
            new_score = func(perturbed_solution)
            
            # Calculate the probability of accepting the new solution
            prob_accept = 1.0
            if new_score < best_score:
                prob_accept = np.exp((best_score - new_score) / 10.0)
            
            # Accept the new solution with probability prob_accept
            if random.random() < prob_accept:
                self.population.append(perturbed_solution)
                self.best_func = perturbed_solution
                self.best_score = new_score
                break
            
            # If the new solution is worse than the current best, revert to the previous solution
            if new_score > best_score:
                self.population[self.t] = self.population[self.t-1]
                self.best_func = self.population[self.t]
                self.best_score = best_score
                break
            
            # Decrease the temperature for simulated annealing
            self.t += 1

# Example usage
noxys = Noxys(1000, 5)
noxys(func)