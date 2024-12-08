import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.population_size = 100
        self.crossover_rate = 0.5
        self.mutation_rate = 0.1
        self.population_history = []

    def __call__(self, func):
        # Evaluate the function with the given budget
        func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], self.population_size)])
        
        # Select the top-performing individuals
        top_individuals = np.argsort(func_evaluations)[-self.population_size:]
        
        # Create a new population by crossover and mutation
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(top_individuals, 2)
            child = (parent1 + parent2) / 2
            if random.random() < self.mutation_rate:
                child = random.uniform(self.search_space[0], self.search_space[1])
            new_population.append(child)
        
        # Replace the old population with the new one
        self.population = new_population
        
        # Evaluate the new population
        new_func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], len(new_population))])
        
        # Return the best individual
        best_individual = np.argmax(new_func_evaluations)
        self.population_history.append((func_evaluations, best_individual))
        
        # Refine the strategy based on the probability 0.45
        if random.random() < 0.45:
            new_individual = self.evaluate_fitness(self.population[best_individual], self.logger)
            self.population[best_individual] = new_individual
        
        return new_population[best_individual]

    def evaluate_fitness(self, individual, logger):
        # Evaluate the function at the individual
        func_value = func(individual)
        
        # Log the fitness value
        l2 = logger.budget / (self.budget + 1)
        logger.trigger.ALWAYS.log(func_value, l2)
        
        return func_value

# Usage
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a logger instance
logger = logging.getLogger(__name__)

# Create a file handler and a stream handler
file_handler = logging.FileHandler('bbob_file.log')
stream_handler = logging.StreamHandler()

# Create a formatter and set the formatter for the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add the handlers to the logger instance
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Create an instance of the BlackBoxOptimizer class
optimizer = BlackBoxOptimizer(budget=100, dim=10)

# Run the optimizer
for _ in range(10):
    func = lambda x: np.sin(x)
    individual = random.uniform(-10, 10)
    best_individual = optimizer(individual)
    print(optimizer.population_history[-1])