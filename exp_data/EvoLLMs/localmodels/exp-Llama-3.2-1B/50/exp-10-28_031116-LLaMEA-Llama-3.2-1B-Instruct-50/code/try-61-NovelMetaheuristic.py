import numpy as np

class NovelMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitness_values = np.zeros(self.population_size)

    def __call__(self, func):
        for _ in range(self.budget):
            # Select an individual from the population
            individual = self.population[np.random.randint(0, self.population_size)]
            
            # Evaluate the function at the individual's bounds
            func_value = func(individual)
            
            # Refine the individual using the selected solution
            refined_individual = individual.copy()
            if np.random.rand() < 0.45:
                # Perturb the individual
                perturbation = np.random.uniform(-1.0, 1.0, self.dim)
                refined_individual += perturbation
                refined_individual = np.clip(refined_individual, -5.0, 5.0)
            
            # Evaluate the refined individual at the bounds
            refined_func_value = func(refined_individual)
            
            # Update the fitness value
            self.fitness_values[np.random.randint(0, self.population_size)] = refined_func_value
        
        # Return the best individual
        return np.argmax(self.fitness_values)

# BBOB test suite functions
def func1(x):
    return np.sum(x**2)

def func2(x):
    return np.prod(x)

def func3(x):
    return np.mean(x)

def func4(x):
    return np.sum(x**3)

def func5(x):
    return np.prod(x)

def func6(x):
    return np.mean(x)

# Run the algorithm
metaheuristic = NovelMetaheuristic(1000, 5)
best_func = metaheuristic(func1)
print(f"Best function: {best_func}")
print(f"Best function value: {func1(best_func)}")