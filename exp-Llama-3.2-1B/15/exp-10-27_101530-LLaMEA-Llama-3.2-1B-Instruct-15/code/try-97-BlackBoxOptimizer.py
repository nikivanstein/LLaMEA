import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Refine the strategy based on the current function evaluations
            if self.func_evaluations < 0.15 * self.budget:
                # Increase the mutation rate to increase exploration
                mutation_rate = 0.01
            else:
                mutation_rate = 0.001
            # Generate a new individual using the current strategy
            new_individual = self.evaluate_fitness(new_individual)
            # Evaluate the new individual
            new_func_value = func(new_individual)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Check if the new individual is within the budget
            if self.func_evaluations < self.budget:
                # If not, return the new individual
                return new_individual
        # If the budget is reached, return the best individual found so far
        return self.search_space[0], self.search_space[1]

def mutation_exp(individual, mutation_rate):
    # Perform mutation on the individual
    mutated_individual = individual + random.uniform(-mutation_rate, mutation_rate)
    return mutated_individual

def evaluate_fitness(individual, func, budget):
    # Evaluate the function at the individual
    func_value = func(individual)
    # Return the fitness value
    return func_value

def generate_random_point(search_space):
    # Generate a random point in the search space
    return (random.uniform(search_space[0], search_space[1]), random.uniform(search_space[0], search_space[1]))

def main():
    # Initialize the BlackBoxOptimizer with a budget and dimension
    optimizer = BlackBoxOptimizer(100, 10)
    # Run the optimization algorithm
    best_individual = optimizer()
    # Save the best individual to a file
    np.save('currentexp/aucs-BlackBoxOptimizer-1.npy', best_individual)
    print("Best individual found:", best_individual)

if __name__ == "__main__":
    main()