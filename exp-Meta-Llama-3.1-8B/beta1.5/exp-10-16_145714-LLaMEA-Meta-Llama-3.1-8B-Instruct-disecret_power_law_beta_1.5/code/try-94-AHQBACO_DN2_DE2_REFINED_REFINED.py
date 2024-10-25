import numpy as np
import random

class AHQBACO_DN2_DE2_REFINED_REFINED:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory = np.random.uniform(-5, 5, (budget, dim))
        self.velocity = np.random.uniform(-1, 1, (budget, dim))
        self.best_solution = np.zeros(dim)
        self.best_fitness = np.inf
        self.exploration_rate = 0.5  # dynamic exploration rate
        self.convergence_threshold = 0.1  # convergence threshold
        self.F = 0.5  # differential evolution parameter
        self.CR = 0.5  # differential evolution parameter
        self.exploration_rate_adjustment = 0.92  # adjustment rate for exploration rate
        self.dynamic_neighborhood_size = 0.1 * budget  # dynamic neighborhood size

    def __call__(self, func):
        convergence = np.zeros(self.budget)
        for i in range(self.budget):
            fitness = func(self.harmony_memory[i])
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = self.harmony_memory[i]
            # Harmony Search
            if random.random() < 0.3:
                self.harmony_memory[i] = self.harmony_memory[i] + np.random.uniform(-1, 1, self.dim)
                self.harmony_memory[i] = np.clip(self.harmony_memory[i], -5, 5)
            # Differential Evolution
            else:
                # Select three random solutions
                index1, index2, index3 = random.sample(range(self.budget), 3)
                # Create a trial solution
                trial_solution = self.harmony_memory[i] + self.F * (self.harmony_memory[index1] - self.harmony_memory[index2]) + self.CR * (self.harmony_memory[index3] - self.harmony_memory[i])
                trial_solution = np.clip(trial_solution, -5, 5)
                # Evaluate the trial solution
                trial_fitness = func(trial_solution)
                # Update the solution if the trial solution is better
                if trial_fitness < func(self.harmony_memory[i]):
                    self.harmony_memory[i] = trial_solution
            convergence[i] = np.linalg.norm(self.harmony_memory[i] - self.best_solution)
            if np.mean(convergence) < self.convergence_threshold:
                self.exploration_rate = max(0.1, self.exploration_rate * 0.8)
            else:
                self.exploration_rate = min(0.9, self.exploration_rate * 1.1)
        return self.best_solution, self.best_fitness

# Example usage:
def sphere(x):
    return np.sum(x**2)

budget = 1000
dim = 10
ahqbaco_dn2_de2_refined_refined = AHQBACO_DN2_DE2_REFINED_REFINED(budget, dim)
best_solution, best_fitness = ahqbaco_dn2_de2_refined_refined(sphere)
print("Best solution:", best_solution)
print("Best fitness:", best_fitness)