import numpy as np
from scipy.optimize import differential_evolution
import os

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

# Probabilistic mutation function
def probabilistic_mutation(individual, mutation_prob):
    if np.random.rand() < mutation_prob:
        # Perform mutation
        new_individual = individual + np.random.uniform(-0.1, 0.1, size=len(individual))
        return new_individual
    else:
        return individual

# Probabilistic selection function
def probabilistic_selection(population, mutation_prob):
    if np.random.rand() < mutation_prob:
        # Perform mutation
        new_population = [probabilistic_mutation(individual, mutation_prob) for individual in population]
        return new_population
    else:
        return population

# Probabilistic crossover function
def probabilistic_crossover(parent1, parent2, crossover_prob):
    if np.random.rand() < crossover_prob:
        # Perform crossover
        child = (parent1 + parent2) / 2
        return child
    else:
        return parent1

# Probabilistic hybrid evolution function
def probabilistic_hybrid_evolution(func, population, mutation_prob, crossover_prob, budget):
    for i in range(budget):
        # Selection
        population = probabilistic_selection(population, mutation_prob)

        # Crossover
        new_population = []
        for j in range(len(population)):
            if np.random.rand() < crossover_prob:
                parent1 = population[j]
                parent2 = population[(j+1) % len(population)]
                child = probabilistic_crossover(parent1, parent2, crossover_prob)
                new_population.append(child)
            else:
                new_population.append(parent1)

        # Mutation
        population = probabilistic_selection(new_population, mutation_prob)

        # Evaluate fitness
        fitness = [func(individual) for individual in population]
        best_individual = np.argmin(fitness)
        population = [population[best_individual]]

    return population[0]

# Example usage:
if __name__ == "__main__":
    # Define a sample black box function
    def func(x):
        return np.sum(x**2)

    # Initialize the probabilistic hybrid evolution algorithm
    phoe = DPHE(budget=100, dim=10)

    # Optimize the function
    result = phoe(func)

    # Print the result
    if result is not None:
        print("Optimal solution:", result)
    else:
        print("Failed to converge")

# Update the current population of algorithms
current_population = [
    {"name": "DPHE", "description": "Novel \"Differential Perturbation and Hybrid Evolution\" algorithm", "score": -np.inf},
    {"name": "DPHE", "description": "Novel \"Differential Perturbation and Hybrid Evolution\" algorithm with probabilistic mutation", "score": -np.inf}
]

selected_algorithm = "DPHE"
current_population[selected_algorithm]["score"] = -np.sum(func(result))

# Update the selected solution
if current_population[selected_algorithm]["score"] > current_population[0]["score"]:
    selected_algorithm = "DPHE"
elif current_population[selected_algorithm]["score"] > current_population[1]["score"]:
    selected_algorithm = "DPHE with probabilistic mutation"

# Update the current population of algorithms
current_population[selected_algorithm]["score"] = -np.sum(func(result))

# Save the updated current population of algorithms
np.save("currentexp/aucs-{}.npy".format(selected_algorithm), current_population)