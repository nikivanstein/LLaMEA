import numpy as np
import random
from scipy.optimize import minimize

class NonLocalTemperatureMetaheuristic:
    def __init__(self, budget, dim, alpha=0.5, mu=0.1, tau=0.9):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.mu = mu
        self.tau = tau
        self.temp = 1.0
        self.best_func = None
        self.history = []

    def __call__(self, func):
        if self.budget <= 0:
            raise ValueError("Budget cannot be zero or negative")

        num_evals = 0
        while num_evals < self.budget and self.best_func is None:
            # Generate a random perturbation
            perturbation = np.random.uniform(-self.dim, self.dim)

            # Evaluate the new function
            new_func = func + perturbation

            # Check if the new function is better
            if np.random.rand() < self.alpha:
                self.best_func = new_func
            else:
                # If the new function is not better, revert the perturbation
                perturbation *= self.tau
                new_func = func + perturbation

            # Update the best function
            self.best_func = new_func

            num_evals += 1

        return self.best_func

    def evolve(self, func, initial_func, budget):
        # Initialize the population
        population = [initial_func]

        for _ in range(self.budget):
            # Select the fittest individual
            fittest_individual = population[-1]

            # Generate a new individual
            new_individual = fittest_individual + np.random.uniform(-self.dim, self.dim)

            # Check if the new individual is within the search space
            if np.random.rand() < self.alpha:
                new_individual = fittest_individual - self.dim
            else:
                # If the new individual is not better, revert the perturbation
                new_individual *= self.tau

            # Evaluate the new individual
            new_func = func(new_individual)

            # Update the population
            population.append(new_individual)

            # Check if the budget is reached
            if len(population) >= budget:
                break

        return population

    def mutate(self, individual):
        # Generate a random mutation
        mutation = np.random.uniform(-self.dim, self.dim)

        # Apply the mutation to the individual
        mutated_individual = individual + mutation

        # Check if the mutated individual is within the search space
        if np.random.rand() < self.alpha:
            mutated_individual = individual - self.dim
        else:
            # If the mutated individual is not better, revert the mutation
            mutated_individual *= self.tau

        return mutated_individual

# One-line description: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# Code: 
# ```python
# NonLocalTemperatureMetaheuristic: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# Description: A novel metaheuristic algorithm that combines non-local temperature and adaptive mutation to optimize black box functions.
# Code: 
# ```python
import numpy as np
import random
from scipy.optimize import minimize

class NonLocalTemperatureMetaheuristic:
    def __init__(self, budget, dim, alpha=0.5, mu=0.1, tau=0.9):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.mu = mu
        self.tau = tau
        self.temp = 1.0
        self.best_func = None
        self.history = []

    def __call__(self, func):
        if self.budget <= 0:
            raise ValueError("Budget cannot be zero or negative")

        num_evals = 0
        while num_evals < self.budget and self.best_func is None:
            # Generate a random perturbation
            perturbation = np.random.uniform(-self.dim, self.dim)

            # Evaluate the new function
            new_func = func + perturbation

            # Check if the new function is better
            if np.random.rand() < self.alpha:
                self.best_func = new_func
            else:
                # If the new function is not better, revert the perturbation
                perturbation *= self.tau
                new_func = func + perturbation

            # Update the best function
            self.best_func = new_func

            num_evals += 1

        return self.best_func

    def evolve(self, func, initial_func, budget):
        # Initialize the population
        population = [initial_func]

        for _ in range(self.budget):
            # Select the fittest individual
            fittest_individual = population[-1]

            # Generate a new individual
            new_individual = fittest_individual + np.random.uniform(-self.dim, self.dim)

            # Check if the new individual is within the search space
            if np.random.rand() < self.alpha:
                new_individual = fittest_individual - self.dim
            else:
                # If the new individual is not better, revert the perturbation
                new_individual *= self.tau

            # Evaluate the new individual
            new_func = func(new_individual)

            # Update the population
            population.append(new_individual)

            # Check if the budget is reached
            if len(population) >= budget:
                break

        return population

    def mutate(self, individual):
        # Generate a random mutation
        mutation = np.random.uniform(-self.dim, self.dim)

        # Apply the mutation to the individual
        mutated_individual = individual + mutation

        # Check if the mutated individual is within the search space
        if np.random.rand() < self.alpha:
            mutated_individual = individual - self.dim
        else:
            # If the mutated individual is not better, revert the mutation
            mutated_individual *= self.tau

        return mutated_individual

# One-line description: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# Code: 
# ```python
# NonLocalTemperatureMetaheuristic: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# Description: A novel metaheuristic algorithm that combines non-local temperature and adaptive mutation to optimize black box functions.
# Code: 
# ```python
import numpy as np
import random

class NonLocalTemperatureMetaheuristic:
    def __init__(self, budget, dim, alpha=0.5, mu=0.1, tau=0.9):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.mu = mu
        self.tau = tau
        self.temp = 1.0
        self.best_func = None
        self.history = []

    def __call__(self, func):
        if self.budget <= 0:
            raise ValueError("Budget cannot be zero or negative")

        num_evals = 0
        while num_evals < self.budget and self.best_func is None:
            # Generate a random perturbation
            perturbation = np.random.uniform(-self.dim, self.dim)

            # Evaluate the new function
            new_func = func + perturbation

            # Check if the new function is better
            if np.random.rand() < self.alpha:
                self.best_func = new_func
            else:
                # If the new function is not better, revert the perturbation
                perturbation *= self.tau
                new_func = func + perturbation

            # Update the best function
            self.best_func = new_func

            num_evals += 1

        return self.best_func

    def evolve(self, func, initial_func, budget):
        # Initialize the population
        population = [initial_func]

        for _ in range(self.budget):
            # Select the fittest individual
            fittest_individual = population[-1]

            # Generate a new individual
            new_individual = fittest_individual + np.random.uniform(-self.dim, self.dim)

            # Check if the new individual is within the search space
            if np.random.rand() < self.alpha:
                new_individual = fittest_individual - self.dim
            else:
                # If the new individual is not better, revert the perturbation
                new_individual *= self.tau

            # Evaluate the new individual
            new_func = func(new_individual)

            # Update the population
            population.append(new_individual)

            # Check if the budget is reached
            if len(population) >= budget:
                break

        return population

    def mutate(self, individual):
        # Generate a random mutation
        mutation = np.random.uniform(-self.dim, self.dim)

        # Apply the mutation to the individual
        mutated_individual = individual + mutation

        # Check if the mutated individual is within the search space
        if np.random.rand() < self.alpha:
            mutated_individual = individual - self.dim
        else:
            # If the mutated individual is not better, revert the mutation
            mutated_individual *= self.tau

        return mutated_individual

# One-line description: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# Code: 
# ```python
# NonLocalTemperatureMetaheuristic: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# Description: A novel metaheuristic algorithm that combines non-local temperature and adaptive mutation to optimize black box functions.
# Code: 
# ```python
import numpy as np
import random

class NonLocalTemperatureMetaheuristic:
    def __init__(self, budget, dim, alpha=0.5, mu=0.1, tau=0.9):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.mu = mu
        self.tau = tau
        self.temp = 1.0
        self.best_func = None
        self.history = []

    def __call__(self, func):
        if self.budget <= 0:
            raise ValueError("Budget cannot be zero or negative")

        num_evals = 0
        while num_evals < self.budget and self.best_func is None:
            # Generate a random perturbation
            perturbation = np.random.uniform(-self.dim, self.dim)

            # Evaluate the new function
            new_func = func + perturbation

            # Check if the new function is better
            if np.random.rand() < self.alpha:
                self.best_func = new_func
            else:
                # If the new function is not better, revert the perturbation
                perturbation *= self.tau
                new_func = func + perturbation

            # Update the best function
            self.best_func = new_func

            num_evals += 1

        return self.best_func

    def evolve(self, func, initial_func, budget):
        # Initialize the population
        population = [initial_func]

        for _ in range(self.budget):
            # Select the fittest individual
            fittest_individual = population[-1]

            # Generate a new individual
            new_individual = fittest_individual + np.random.uniform(-self.dim, self.dim)

            # Check if the new individual is within the search space
            if np.random.rand() < self.alpha:
                new_individual = fittest_individual - self.dim
            else:
                # If the new individual is not better, revert the perturbation
                new_individual *= self.tau

            # Evaluate the new individual
            new_func = func(new_individual)

            # Update the population
            population.append(new_individual)

            # Check if the budget is reached
            if len(population) >= budget:
                break

        return population

    def mutate(self, individual):
        # Generate a random mutation
        mutation = np.random.uniform(-self.dim, self.dim)

        # Apply the mutation to the individual
        mutated_individual = individual + mutation

        # Check if the mutated individual is within the search space
        if np.random.rand() < self.alpha:
            mutated_individual = individual - self.dim
        else:
            # If the mutated individual is not better, revert the mutation
            mutated_individual *= self.tau

        return mutated_individual

# One-line description: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# Code: 
# ```python
# NonLocalTemperatureMetaheuristic: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# Description: A novel metaheuristic algorithm that combines non-local temperature and adaptive mutation to optimize black box functions.
# Code: 
# ```python
import numpy as np
import random

class NonLocalTemperatureMetaheuristic:
    def __init__(self, budget, dim, alpha=0.5, mu=0.1, tau=0.9):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.mu = mu
        self.tau = tau
        self.temp = 1.0
        self.best_func = None
        self.history = []

    def __call__(self, func):
        if self.budget <= 0:
            raise ValueError("Budget cannot be zero or negative")

        num_evals = 0
        while num_evals < self.budget and self.best_func is None:
            # Generate a random perturbation
            perturbation = np.random.uniform(-self.dim, self.dim)

            # Evaluate the new function
            new_func = func + perturbation

            # Check if the new function is better
            if np.random.rand() < self.alpha:
                self.best_func = new_func
            else:
                # If the new function is not better, revert the perturbation
                perturbation *= self.tau
                new_func = func + perturbation

            # Update the best function
            self.best_func = new_func

            num_evals += 1

        return self.best_func

    def evolve(self, func, initial_func, budget):
        # Initialize the population
        population = [initial_func]

        for _ in range(self.budget):
            # Select the fittest individual
            fittest_individual = population[-1]

            # Generate a new individual
            new_individual = fittest_individual + np.random.uniform(-self.dim, self.dim)

            # Check if the new individual is within the search space
            if np.random.rand() < self.alpha:
                new_individual = fittest_individual - self.dim
            else:
                # If the new individual is not better, revert the perturbation
                new_individual *= self.tau

            # Evaluate the new individual
            new_func = func(new_individual)

            # Update the population
            population.append(new_individual)

            # Check if the budget is reached
            if len(population) >= budget:
                break

        return population

    def mutate(self, individual):
        # Generate a random mutation
        mutation = np.random.uniform(-self.dim, self.dim)

        # Apply the mutation to the individual
        mutated_individual = individual + mutation

        # Check if the mutated individual is within the search space
        if np.random.rand() < self.alpha:
            mutated_individual = individual - self.dim
        else:
            # If the mutated individual is not better, revert the mutation
            mutated_individual *= self.tau

        return mutated_individual

# One-line description: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# Code: 
# ```python
# NonLocalTemperatureMetaheuristic: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# Description: A novel metaheuristic algorithm that combines non-local temperature and adaptive mutation to optimize black box functions.
# Code: 
# ```python
import numpy as np
import random

class NonLocalTemperatureMetaheuristic:
    def __init__(self, budget, dim, alpha=0.5, mu=0.1, tau=0.9):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.mu = mu
        self.tau = tau
        self.temp = 1.0
        self.best_func = None
        self.history = []

    def __call__(self, func):
        if self.budget <= 0:
            raise ValueError("Budget cannot be zero or negative")

        num_evals = 0
        while num_evals < self.budget and self.best_func is None:
            # Generate a random perturbation
            perturbation = np.random.uniform(-self.dim, self.dim)

            # Evaluate the new function
            new_func = func + perturbation

            # Check if the new function is better
            if np.random.rand() < self.alpha:
                self.best_func = new_func
            else:
                # If the new function is not better, revert the perturbation
                perturbation *= self.tau
                new_func = func + perturbation

            # Update the best function
            self.best_func = new_func

            num_evals += 1

        return self.best_func

    def evolve(self, func, initial_func, budget):
        # Initialize the population
        population = [initial_func]

        for _ in range(self.budget):
            # Select the fittest individual
            fittest_individual = population[-1]

            # Generate a new individual
            new_individual = fittest_individual + np.random.uniform(-self.dim, self.dim)

            # Check if the new individual is within the search space
            if np.random.rand() < self.alpha:
                new_individual = fittest_individual - self.dim
            else:
                # If the new individual is not better, revert the perturbation
                new_individual *= self.tau

            # Evaluate the new individual
            new_func = func(new_individual)

            # Update the population
            population.append(new_individual)

            # Check if the budget is reached
            if len(population) >= budget:
                break

        return population

    def mutate(self, individual):
        # Generate a random mutation
        mutation = np.random.uniform(-self.dim, self.dim)

        # Apply the mutation to the individual
        mutated_individual = individual + mutation

        # Check if the mutated individual is within the search space
        if np.random.rand() < self.alpha:
            mutated_individual = individual - self.dim
        else:
            # If the mutated individual is not better, revert the mutation
            mutated_individual *= self.tau

        return mutated_individual