import numpy as np

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
        
        def evaluate_fitness(individual):
            y = inner(x)
            return y
        
        # Initialize population with random individuals
        population = np.array([inner(x) for x in np.random.uniform(-5.0, 5.0, self.dim)])
        
        while len(population) < self.budget:
            # Evaluate the function at the current population
            fitness_values = evaluate_fitness(population)
            
            # Select the fittest individual
            fittest_individual = population[np.argmax(fitness_values)]
            
            # Refine the strategy by changing the direction of the fittest individual
            new_individual = self.refine_strategy(fittest_individual, population, func)
            
            # Evaluate the new individual
            new_fitness = evaluate_fitness(new_individual)
            
            # If we've reached the maximum number of iterations, stop refining
            if new_fitness == fittest_individual:
                break
            
            # Add the new individual to the population
            population = np.append(population, new_individual)
        
        return population

    def refine_strategy(self, individual, population, func):
        # Calculate the gradient of the function at the individual
        gradient = np.gradient(func(individual), axis=0)
        
        # Calculate the direction of the individual
        direction = gradient / np.linalg.norm(gradient)
        
        # Normalize the direction
        direction = direction / np.linalg.norm(direction)
        
        # Refine the individual by adding a small noise to the direction
        new_individual = individual + np.random.normal(0, 0.1, self.dim)
        
        # Refine the strategy by changing the direction of the new individual
        new_individual = self.refine_strategy(new_individual, population, func)
        
        return new_individual

# One-line description with the main idea
# Novel Metaheuristic Algorithm for Black Box Optimization
# Refines the solution by iteratively refining the direction of the fittest individual using gradient descent and noise injection.