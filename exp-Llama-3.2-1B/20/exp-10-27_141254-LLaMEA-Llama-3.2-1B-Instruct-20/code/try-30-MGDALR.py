import numpy as np

class MGDALR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.learning_rate = 0.01
        self.explore_count = 0
        self.max_explore_count = budget
        self.learning_rate_adaptation = False

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
            if not self.learning_rate_adaptation:
                learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
            else:
                learning_rate = self.learning_rate * np.exp(-self.explore_count / 1000)
            dx = -np.dot(x - inner(x), np.gradient(y))
            x += learning_rate * dx
            
            # Update the exploration count
            self.explore_count += 1
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
        
        return x

class Individual:
    def __init__(self, x):
        self.x = x
    
    def __call__(self, func):
        return inner(self.x)

class Gradient:
    def __init__(self, func, learning_rate):
        self.func = func
        self.learning_rate = learning_rate
    
    def __call__(self, x):
        return self.func(x)

def evaluate_fitness(individual, func, logger):
    new_individual = individual()
    new_individual.func = Gradient(func, logger)
    return new_individual

def main():
    # Initialize the MGDALR algorithm
    mgdalr = MGDALR(100, 10)
    
    # Define the evaluation function
    def evaluate_func(individual):
        return individual.func(individual.x)
    
    # Define the logger
    logger = np.random.rand(1)
    
    # Initialize the individual
    individual = Individual(np.array([-5.0] * 10))
    
    # Initialize the best solution
    best_solution = Individual(np.array([-5.0] * 10))
    best_score = -np.inf
    
    # Evaluate the function 100 times
    for _ in range(100):
        # Evaluate the function
        score = evaluate_func(individual)
        
        # Update the best solution if necessary
        if score > best_score:
            best_solution = individual
            best_score = score
        
        # Update the individual
        individual = evaluate_fitness(individual, evaluate_func, logger)
        
        # Update the logger
        logger = np.random.rand(1)
    
    # Print the best solution
    print("Best solution:", best_solution.x)
    print("Best score:", best_score)

if __name__ == "__main__":
    main()