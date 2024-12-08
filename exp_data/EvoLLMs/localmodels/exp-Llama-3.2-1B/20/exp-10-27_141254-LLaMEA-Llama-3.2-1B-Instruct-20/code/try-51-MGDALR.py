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

    def update_individual(self, individual):
        # Refine the strategy using a novel heuristic
        # Generate a new individual by perturbing the current one
        # with a probability proportional to the fitness value
        new_individual = individual
        if np.random.rand() < 0.2:
            # Perturb the current individual
            perturbation = np.random.normal(0, 1, self.dim)
            new_individual = individual + perturbation
        
        # Evaluate the new individual
        new_fitness = self.func(new_individual)
        
        # Refine the strategy based on the fitness value
        if new_fitness > individual.f:
            # If the new individual has a higher fitness value, update the individual
            new_individual = individual + 0.1 * (new_individual - individual)
        elif new_fitness < individual.f:
            # If the new individual has a lower fitness value, update the individual
            new_individual = individual - 0.1 * (individual - new_individual)
        
        # Update the individual's fitness value
        individual.f = new_fitness
        
        # Update the individual's strategy
        if np.random.rand() < 0.2:
            # Perturb the individual's strategy with a probability proportional to the fitness value
            perturbation = np.random.normal(0, 1, self.dim)
            individual.strategy = individual.strategy + perturbation
        
        # Update the individual's strategy based on the fitness value
        if individual.f > 0.5:
            # If the individual's fitness value is above a threshold, update the individual's strategy
            individual.strategy = individual.strategy + 0.1 * (individual.strategy - individual.strategy)
        elif individual.f < 0.5:
            # If the individual's fitness value is below a threshold, update the individual's strategy
            individual.strategy = individual.strategy - 0.1 * (individual.strategy - individual.strategy)
        
        # Return the updated individual
        return new_individual

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 