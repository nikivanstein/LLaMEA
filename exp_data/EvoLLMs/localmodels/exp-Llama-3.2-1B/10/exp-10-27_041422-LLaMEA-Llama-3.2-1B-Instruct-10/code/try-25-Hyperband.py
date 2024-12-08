import numpy as np

class Hyperband:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_func = None
        self.best_func_evals = 0
        self.sample_size = 1
        self.sample_dir = None
        self.num_evals = 0

    def __call__(self, func):
        if self.budget <= 0:
            raise ValueError("Budget cannot be zero or negative")
        
        if self.best_func is not None:
            return self.best_func
        
        # Initialize the best function and its evaluation count
        self.best_func = func
        self.best_func_evals = 1
        
        # Set the sample size and directory
        self.sample_size = 10
        self.sample_dir = f"sample_{self.sample_size}"
        
        # Perform adaptive sampling
        for _ in range(self.budget):
            # Generate a random sample of size self.sample_size
            sample = np.random.uniform(-5.0, 5.0, size=self.dim)
            
            # Evaluate the function at the current sample
            func_eval = func(sample)
            
            # If this is the first evaluation, update the best function
            if self.best_func_evals == 1:
                self.best_func = func_eval
                self.best_func_evals = 1
            # Otherwise, update the best function if the current evaluation is better
            else:
                if func_eval > self.best_func:
                    self.best_func = func_eval
                    self.best_func_evals = 1
                else:
                    self.best_func_evals += 1
            
            # Save the current sample to the sample directory
            np.save(f"{self.sample_dir}_{self.sample_size}_{_}", sample)
        
        # Update the population with the best function
        self.best_func = func
        self.num_evals += 1
        return self.best_func

def __call__(self, func, population):
    best_func = None
    best_func_evals = 0
    for individual in population:
        func_eval = func(individual)
        if func_eval > best_func_evals:
            best_func = individual
            best_func_evals = func_eval
    self.best_func = best_func
    return self.best_func

def evaluate_fitness(individual, func, budget, population, logger):
    # Implement the fitness function evaluation here
    # For simplicity, assume it's a black box function
    return func(individual)

# Test the algorithm
func = lambda x: x**2
budget = 10
population = [np.random.uniform(-5.0, 5.0, size=10) for _ in range(100)]
hyperband = Hyperband(budget, 10)
best_func = None
for _ in range(10):
    best_func = __call__(func, population)
    print(f"Best function: {best_func}")
    logger.info(f"Best fitness: {evaluate_fitness(best_func, func, budget, population, logger)}")