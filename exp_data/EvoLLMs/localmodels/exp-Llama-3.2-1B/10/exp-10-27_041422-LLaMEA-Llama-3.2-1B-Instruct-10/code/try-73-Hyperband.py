import numpy as np
from scipy.optimize import differential_evolution

class Hyperband:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_func = None
        self.best_func_evals = 0
        self.sample_size = 10
        self.sample_dir = f"sample_{self.sample_size}"
        self.fitness_history = []

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
        
        # Store the fitness history
        self.fitness_history.append(func_eval)
        
        return self.best_func

# Example usage:
if __name__ == "__main__":
    # Define a black box function
    def func(x):
        return np.sin(x)
    
    # Create an instance of Hyperband with a budget of 1000 and dimension 2
    hyperband = Hyperband(1000, 2)
    
    # Optimize the function using Hyperband
    new_individual = hyperband.func(func)
    
    # Print the fitness history
    print("Fitness History:", hyperband.fitness_history)
    
    # Plot the fitness history
    import matplotlib.pyplot as plt
    plt.plot(hyperband.fitness_history)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title("Fitness History")
    plt.show()