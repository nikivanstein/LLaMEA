import numpy as np
import random
from copy import deepcopy

class NonLocalTemperatureMetaheuristic:
    def __init__(self, budget, dim, alpha=0.5, mu=0.1, tau=0.9):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.mu = mu
        self.tau = tau
        self.temp = 1.0
        self.best_func = None
        self.population = []

    def __call__(self, func):
        if self.budget <= 0:
            raise ValueError("Budget cannot be zero or negative")

        num_evals = 0
        while num_evals < self.budget and self.best_func is None:
            # Generate a random perturbation
            perturbation = np.random.uniform(-self.dim, self.dim)

            # Evaluate the new function
            new_func = func + perturbation

            # Check if the new function is better
            if np.random.rand() < self.alpha:
                self.best_func = new_func
            else:
                # If the new function is not better, revert the perturbation
                perturbation *= self.tau
                new_func = func + perturbation

            # Clone the current population and add the new individual
            self.population.append(deepcopy(self.best_func))

            num_evals += 1

        return self.best_func

# One-line description: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# Code: 
# ```python
# import numpy as np
# import random

class NonLocalTemperatureMetaheuristic:
    def __init__(self, budget, dim, alpha=0.5, mu=0.1, tau=0.9):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.mu = mu
        self.tau = tau
        self.temp = 1.0
        self.best_func = None
        self.population = []

    def __call__(self, func):
        if self.budget <= 0:
            raise ValueError("Budget cannot be zero or negative")

        num_evals = 0
        while num_evals < self.budget and self.best_func is None:
            # Generate a random perturbation
            perturbation = np.random.uniform(-self.dim, self.dim)

            # Evaluate the new function
            new_func = func + perturbation

            # Check if the new function is better
            if np.random.rand() < self.alpha:
                self.best_func = new_func
            else:
                # If the new function is not better, revert the perturbation
                perturbation *= self.tau
                new_func = func + perturbation

            # Clone the current population and add the new individual
            self.population.append(deepcopy(new_func))

            num_evals += 1

        return self.best_func

# Function to evaluate the fitness of a function
def evaluate_fitness(func, budget):
    for _ in range(budget):
        func()
    return func

# Function to generate a new individual
def generate_individual(func, dim):
    return func()

# Function to mutate an individual
def mutate(individual):
    return individual + np.random.uniform(-1, 1)

# Function to update the population
def update_population(population, func):
    for i in range(len(population)):
        new_individual = generate_individual(func, len(population))
        population[i] = mutate(new_individual)
    return population

# Main function
def main():
    # Set the parameters
    budget = 1000
    dim = 10
    alpha = 0.5
    mu = 0.1
    tau = 0.9

    # Create an instance of the metaheuristic
    metaheuristic = NonLocalTemperatureMetaheuristic(budget, dim, alpha, mu, tau)

    # Evaluate the fitness of the function
    func = lambda: np.sin(np.linspace(0, 2*np.pi, 100))
    fitness = evaluate_fitness(func, budget)

    # Update the population
    population = update_population(metaheuristic.population, func)

    # Print the best individual
    best_individual = max(population, key=fitness)
    print("Best individual:", best_individual)

if __name__ == "__main__":
    main()