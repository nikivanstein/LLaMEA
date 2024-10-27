import numpy as np
from scipy.optimize import differential_evolution

class MGDALR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.learning_rate = 0.01
        self.explore_count = 0
        self.max_explore_count = budget

    def __call__(self, func):
        def inner(x):
            return func(x)
        
        # Initialize x to the lower bound
        x = np.array([-5.0] * self.dim)
        
        for _ in range(self.budget):
            # Evaluate the function at the current x
            y = inner(x)
            
            # If we've reached the maximum number of iterations, stop exploring
            if self.explore_count >= self.max_explore_count:
                break
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
            
            # Learn a new direction using gradient descent
            learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
            dx = -np.dot(x - inner(x), np.gradient(y))
            x += learning_rate * dx
            
            # Update the exploration count
            self.explore_count += 1
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
        
        return x

    def adaptive_probabilistic_search(self, func, bounds, max_evals):
        # Initialize a population of random solutions
        population = self.generate_initial_population(bounds, max_evals)
        
        # Run the evolution for a fixed number of generations
        for _ in range(100):
            # Evaluate the fitness of each individual in the population
            fitnesses = np.array([func(individual) for individual in population])
            
            # Select the fittest individuals
            fittest_individuals = np.argsort(fitnesses)[::-1][:self.budget]
            
            # Create a new population by adapting the fittest individuals
            new_population = []
            for _ in range(self.budget):
                # Select a random individual from the fittest individuals
                individual = fittest_individuals[np.random.randint(0, len(fittest_individuals))]
                
                # Refine the strategy using probabilistic search
                prob = np.random.rand()
                if prob < 0.2:
                    # Learn a new direction using gradient descent
                    learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
                    dx = -np.dot(individual - func(individual), np.gradient(func(individual)))
                    new_individual = individual + learning_rate * dx
                else:
                    # Use a fixed direction
                    new_individual = individual
                
                # Add the new individual to the new population
                new_population.append(new_individual)
            
            # Update the population
            population = new_population
        
        # Return the best individual in the population
        return np.array(population[np.argmin(fitnesses)])

# Example usage:
func = lambda x: np.sin(x)
bounds = [-5.0, 5.0]
max_evals = 1000
best_solution = MGDALR(1000, 10).adaptive_probabilistic_search(func, bounds, max_evals)
print(best_solution)