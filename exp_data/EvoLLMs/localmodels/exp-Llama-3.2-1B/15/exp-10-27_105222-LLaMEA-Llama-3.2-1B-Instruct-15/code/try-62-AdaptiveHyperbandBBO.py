import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class AdaptiveHyperbandBBO:
    def __init__(self, budget, dim, learning_rate, exploration_rate, noise):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.search_space_dim = self.dim
        self.search_space_bounds = self._set_search_space_bounds()
        self.search_space_bounds_dim = self.dim
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.noise = noise

    def _set_search_space_bounds(self):
        # Set the bounds for the search space based on the noise level
        # and the dimensionality
        if self.noise < 0.1:
            return (-5.0, 5.0)
        else:
            return (-2.0, 2.0)

    def _get_new_point(self, current_point, current_fitness):
        # Use a Gaussian distribution to sample a new point in the search space
        # with a probability proportional to the current fitness
        new_point = np.random.uniform(*self.search_space_bounds, size=self.search_space_bounds_dim)
        new_fitness = current_fitness + np.random.normal(0, 1, size=self.search_space_bounds_dim)
        return new_point, new_fitness

    def __call__(self, func):
        # Initialize the population with random points in the search space
        population = [self._get_new_point(np.random.uniform(*self.search_space_bounds, size=self.search_space_bounds_dim),
                                            func(np.random.uniform(*self.search_space_bounds, size=self.search_space_bounds_dim)))) for _ in range(100)]

        # Evaluate the fitness of each point in the population
        fitnesses = [func(point) for point in population]

        # Select the fittest points based on the fitness values
        fittest_points = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)[:self.budget]

        # Refine the population using Bayesian optimization
        refined_population = []
        for point, fitness in fittest_points:
            # Sample a new point in the search space using Gaussian distribution
            new_point, new_fitness = self._get_new_point(point, fitness)
            # Evaluate the function at the new point
            new_fitness_value = func(new_point)
            # Store the new point in the refined population
            refined_population.append((new_point, new_fitness_value))
            # Update the fitness value and the new fitness
            fitnesses.append(new_fitness_value)

        # Evaluate the fitness of the refined population
        refined_fitnesses = [func(point) for point, fitness in refined_population]

        # Select the fittest points based on the refined fitness values
        fittest_points = sorted(zip(refined_population, refined_fitnesses), key=lambda x: x[1], reverse=True)[:self.budget]

        # Refine the population using hyperband
        refined_population = []
        for point, fitness in fittest_points:
            # Sample a new point in the search space using Gaussian distribution
            new_point, new_fitness = self._get_new_point(point, fitness)
            # Evaluate the function at the new point
            new_fitness_value = func(new_point)
            # Store the new point in the refined population
            refined_population.append((new_point, new_fitness_value))
            # Update the fitness value and the new fitness
            fitnesses.append(new_fitness_value)

        # Evaluate the fitness of the refined population
        refined_fitnesses = [func(point) for point, fitness in refined_population]

        # Select the fittest points based on the refined fitness values
        fittest_points = sorted(zip(refined_population, refined_fitnesses), key=lambda x: x[1], reverse=True)[:self.budget]

        # Refine the population using hyperband again
        refined_population = []
        for point, fitness in fittest_points:
            # Sample a new point in the search space using Gaussian distribution
            new_point, new_fitness = self._get_new_point(point, fitness)
            # Evaluate the function at the new point
            new_fitness_value = func(new_point)
            # Store the new point in the refined population
            refined_population.append((new_point, new_fitness_value))
            # Update the fitness value and the new fitness
            fitnesses.append(new_fitness_value)

        # Evaluate the fitness of the refined population
        refined_fitnesses = [func(point) for point, fitness in refined_population]

        # Select the fittest points based on the refined fitness values
        fittest_points = sorted(zip(refined_population, refined_fitnesses), key=lambda x: x[1], reverse=True)[:self.budget]

        # Refine the population using Bayesian optimization again
        refined_population = []
        for point, fitness in fittest_points:
            # Sample a new point in the search space using Gaussian distribution
            new_point, new_fitness = self._get_new_point(point, fitness)
            # Evaluate the function at the new point
            new_fitness_value = func(new_point)
            # Store the new point in the refined population
            refined_population.append((new_point, new_fitness_value))
            # Update the fitness value and the new fitness
            fitnesses.append(new_fitness_value)

        # Evaluate the fitness of the refined population
        refined_fitnesses = [func(point) for point, fitness in refined_population]

        # Select the fittest points based on the refined fitness values
        fittest_points = sorted(zip(refined_population, refined_fitnesses), key=lambda x: x[1], reverse=True)[:self.budget]

        # Refine the population using hyperband one last time
        refined_population = []
        for point, fitness in fittest_points:
            # Sample a new point in the search space using Gaussian distribution
            new_point, new_fitness = self._get_new_point(point, fitness)
            # Evaluate the function at the new point
            new_fitness_value = func(new_point)
            # Store the new point in the refined population
            refined_population.append((new_point, new_fitness_value))
            # Update the fitness value and the new fitness
            fitnesses.append(new_fitness_value)

        # Evaluate the fitness of the refined population
        refined_fitnesses = [func(point) for point, fitness in refined_population]

        # Select the fittest points based on the refined fitness values
        fittest_points = sorted(zip(refined_population, refined_fitnesses), key=lambda x: x[1], reverse=True)[:self.budget]

        # Return the fittest point
        return fittest_points[0][0], refined_fitnesses[0]

# Example usage:
def test_func1(x):
    return np.sin(x)

def test_func2(x):
    return x**2 + 2*x + 1

hyperband = AdaptiveHyperbandBBO(budget=100, dim=10, learning_rate=0.01, exploration_rate=0.01, noise=0.1)
optimized_func1 = hyperband(test_func1)
optimized_func2 = hyperband(test_func2)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot([optimized_func1, optimized_func2], label=['Test Function 1', 'Test Function 2'])
plt.xlabel('Optimized Function Value')
plt.ylabel('Dimensionality')
plt.title('Black Box Optimization using Adaptive Hyperband and Bayesian Optimization')
plt.legend()
plt.show()