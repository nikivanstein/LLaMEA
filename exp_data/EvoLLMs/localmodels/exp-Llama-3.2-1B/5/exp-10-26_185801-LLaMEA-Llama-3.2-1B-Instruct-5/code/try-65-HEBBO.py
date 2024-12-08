import numpy as np
import random

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population = None
        self.logger = None

    def __call__(self, func):
        if self.population is None:
            self.population = self.initialize_population(func)
        while self.func_evaluations < self.budget:
            func_value = self.evaluate_fitness(self.population)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
            self.population = self.select_next_individual(self.population, func_value)
            self.func_evaluations += 1
        return func_value

    def initialize_population(self, func):
        return np.random.uniform(self.search_space, size=(self.dim,))

    def select_next_individual(self, population, func_value):
        # Novel heuristic algorithm: "Gradient Descent with Local Search"
        # Description: This algorithm uses a combination of gradient descent and local search to refine the solution
        # Code: 
        # ```python
        # Initialize the population with a random starting point
        new_individual = self.initialize_population(func)
        
        # Calculate the gradient of the function at the current population
        gradient = np.gradient(func(new_individual), axis=0)
        
        # Perform a local search to refine the solution
        for _ in range(100):
            # Randomly select a direction in the search space
            direction = np.random.choice([-1, 1], size=self.dim)
            # Update the new individual in the search space
            new_individual += direction * gradient
        
        # Evaluate the new individual using the function
        new_individual_value = func(new_individual)
        
        # If the new individual value is better than the current one, update it
        if new_individual_value > func_value:
            return new_individual
        else:
            return new_individual_value

    def mutate(self, individual):
        # Novel heuristic algorithm: "Mutation with Genetic Algorithm"
        # Description: This algorithm uses a combination of mutation and genetic algorithm to refine the solution
        # Code: 
        # ```python
        # Initialize the population with a random starting point
        new_individual = self.initialize_population(individual)
        
        # Calculate the mutation rate
        mutation_rate = 0.01
        
        # Perform mutation on the new individual
        for i in range(self.dim):
            if random.random() < mutation_rate:
                # Randomly select a gene to mutate
                gene = np.random.choice(self.dim)
                # Mutate the gene
                new_individual[gene] += random.uniform(-1, 1)
        
        # Evaluate the new individual using the function
        new_individual_value = func(new_individual)
        
        # If the new individual value is better than the current one, update it
        if new_individual_value > func_value:
            return new_individual
        else:
            return new_individual_value

# Description: "Gradient Descent with Local Search"
# Code: 
# ```python
# import numpy as np
# import random

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population = None
        self.logger = None

    def __call__(self, func):
        if self.population is None:
            self.population = self.initialize_population(func)
        while self.func_evaluations < self.budget:
            func_value = self.evaluate_fitness(self.population)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
            self.population = self.select_next_individual(self.population, func_value)
            self.func_evaluations += 1
        return func_value

class HESO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population = None
        self.logger = None

    def __call__(self, func):
        if self.population is None:
            self.population = self.initialize_population(func)
        while self.func_evaluations < self.budget:
            func_value = self.evaluate_fitness(self.population)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
            self.population = self.select_next_individual(self.population, func_value)
            self.func_evaluations += 1
        return func_value

class HESO2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population = None
        self.logger = None

    def __call__(self, func):
        if self.population is None:
            self.population = self.initialize_population(func)
        while self.func_evaluations < self.budget:
            func_value = self.evaluate_fitness(self.population)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
            self.population = self.select_next_individual(self.population, func_value)
            self.func_evaluations += 1
        return func_value

# Description: "Gradient Descent with Local Search"
# Code: 
# ```python
# import numpy as np
# import random

class HESO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population = None
        self.logger = None

    def __call__(self, func):
        if self.population is None:
            self.population = self.initialize_population(func)
        while self.func_evaluations < self.budget:
            func_value = self.evaluate_fitness(self.population)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
            self.population = self.select_next_individual(self.population, func_value)
            self.func_evaluations += 1
        return func_value

class HESO2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population = None
        self.logger = None

    def __call__(self, func):
        if self.population is None:
            self.population = self.initialize_population(func)
        while self.func_evaluations < self.budget:
            func_value = self.evaluate_fitness(self.population)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
            self.population = self.select_next_individual(self.population, func_value)
            self.func_evaluations += 1
        return func_value

# Description: "Mutation with Genetic Algorithm"
# Code: 
# ```python
# import numpy as np
# import random

class HESO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population = None
        self.logger = None

    def __call__(self, func):
        if self.population is None:
            self.population = self.initialize_population(func)
        while self.func_evaluations < self.budget:
            func_value = self.evaluate_fitness(self.population)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
            new_individual = self.mutate(self.population)
            self.population = new_individual
            self.func_evaluations += 1
        return func_value

class HESO2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population = None
        self.logger = None

    def __call__(self, func):
        if self.population is None:
            self.population = self.initialize_population(func)
        while self.func_evaluations < self.budget:
            func_value = self.evaluate_fitness(self.population)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
            new_individual = self.mutate(self.population)
            self.population = new_individual
            self.func_evaluations += 1
        return func_value

# Description: "Mutation with Genetic Algorithm"
# Code: 
# ```python
# import numpy as np
# import random

class HESO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population = None
        self.logger = None

    def __call__(self, func):
        if self.population is None:
            self.population = self.initialize_population(func)
        while self.func_evaluations < self.budget:
            func_value = self.evaluate_fitness(self.population)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
            new_individual = self.mutate(self.population)
            self.population = new_individual
            self.func_evaluations += 1
        return func_value

class HESO2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population = None
        self.logger = None

    def __call__(self, func):
        if self.population is None:
            self.population = self.initialize_population(func)
        while self.func_evaluations < self.budget:
            func_value = self.evaluate_fitness(self.population)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
            new_individual = self.mutate(self.population)
            self.population = new_individual
            self.func_evaluations += 1
        return func_value

# Description: "Genetic Algorithm with Mutation"
# Code: 
# ```python
# import numpy as np
# import random

class HESO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population = None
        self.logger = None

    def __call__(self, func):
        if self.population is None:
            self.population = self.initialize_population(func)
        while self.func_evaluations < self.budget:
            func_value = self.evaluate_fitness(self.population)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
            new_individual = self.mutate(self.population)
            self.population = new_individual
            self.func_evaluations += 1
        return func_value

class HESO2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population = None
        self.logger = None

    def __call__(self, func):
        if self.population is None:
            self.population = self.initialize_population(func)
        while self.func_evaluations < self.budget:
            func_value = self.evaluate_fitness(self.population)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
            new_individual = self.mutate(self.population)
            self.population = new_individual
            self.func_evaluations += 1
        return func_value