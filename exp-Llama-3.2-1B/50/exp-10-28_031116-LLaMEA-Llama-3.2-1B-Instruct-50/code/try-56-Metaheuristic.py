import numpy as np

class Metaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []
        self.fitness_values = []

    def __call__(self, func):
        # Initialize the population with random solutions
        for _ in range(self.budget):
            # Generate a random solution in the search space
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            # Evaluate the function at the solution
            fitness = func(solution)
            # Store the fitness value
            self.fitness_values.append(fitness)
            # Store the solution in the population
            self.population.append(solution)

        # Select the fittest solutions
        self.population.sort(key=self.fitness_values.pop, reverse=True)
        # Refine the strategy by changing the individual lines
        for _ in range(10):
            # Select the two fittest solutions
            fittest1 = self.population[0]
            fittest2 = self.population[1]
            # Refine the lower bound
            lower_bound = fittest1
            for dim in range(self.dim):
                if fittest1[dim] < -5.0:
                    lower_bound[dim] = -5.0
                elif fittest1[dim] > 5.0:
                    lower_bound[dim] = 5.0
            # Refine the upper bound
            upper_bound = fittest2
            for dim in range(self.dim):
                if fittest2[dim] < -5.0:
                    upper_bound[dim] = -5.0
                elif fittest2[dim] > 5.0:
                    upper_bound[dim] = 5.0
            # Refine the search space
            for dim in range(self.dim):
                if lower_bound[dim] > upper_bound[dim]:
                    lower_bound[dim] = upper_bound[dim]
                elif lower_bound[dim] < upper_bound[dim]:
                    upper_bound[dim] = lower_bound[dim]

        # Return the best solution
        return lower_bound

# Test the algorithm
def black_box_function(x):
    return x**2 + 2*x + 1

metaheuristic = Metaheuristic(100, 10)
best_solution = metaheuristic(__call__, black_box_function)
print("Best solution:", best_solution)
print("Fitness value:", black_box_function(best_solution))