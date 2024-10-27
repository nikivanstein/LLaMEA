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

    # Refine the strategy with probability-based mutation
    def refine_strategy(individual):
        if np.random.rand() < 0.45:
            # Perturb the individual
            perturbation = np.random.uniform(-1.0, 1.0, size=individual.shape)
            individual += perturbation
            # Clip the individual to the bounds
            individual = np.clip(individual, self.lower_bound, self.upper_bound)
        return individual

    # Update the population with refined strategies
    def update_population(population):
        new_population = []
        for individual in population:
            refined_individual = refine_strategy(individual)
            new_population.append(refined_individual)
        return new_population

    # Evaluate the fitness of the refined population
    def evaluate_fitness(population):
        # Evaluate the fitness of each individual in the population
        fitnesses = []
        for individual in population:
            fitness = func(individual)
            fitnesses.append(fitness)
        return fitnesses

    # Main loop
    population = np.random.uniform(self.lower_bound, self.upper_bound, size=(100, self.dim))
    for i in range(100):
        population = update_population(population)
        fitnesses = evaluate_fitness(population)
        # Select the fittest individuals
        fittest_individuals = population[np.argsort(fitnesses)]
        # Update the population with the fittest individuals
        population = fittest_individuals[:50]