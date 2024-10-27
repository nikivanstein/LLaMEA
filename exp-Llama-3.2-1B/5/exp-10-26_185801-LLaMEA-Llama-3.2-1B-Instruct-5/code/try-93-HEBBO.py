import numpy as np

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population = None

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

    def select_solution(self):
        # Select a random solution from the current population
        self.population = np.random.choice(self.search_space, size=self.budget, replace=False)

    def mutate(self, individual):
        # Randomly swap two individuals in the population
        index1, index2 = np.random.choice(self.budget, size=2, replace=False)
        individual1, individual2 = self.population[index1], self.population[index2]
        self.population[index1], self.population[index2] = individual2, individual1
        return individual1, individual2

    def crossover(self, parent1, parent2):
        # Select a random crossover point and combine the parents
        crossover_point = np.random.randint(1, self.budget)
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        return child

    def evaluate_fitness(self, individual, logger):
        # Evaluate the fitness of the individual using the given function
        func_value = self.func_evaluations(individual)
        if np.isnan(func_value) or np.isinf(func_value):
            raise ValueError("Invalid function value")
        if func_value < 0 or func_value > 1:
            raise ValueError("Function value must be between 0 and 1")
        logger.update(individual, func_value)
        return func_value

    def run(self, func, budget, dim):
        # Initialize the population and select the first solution
        self.select_solution()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler('log.txt')
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Run the evolutionary algorithm
        for _ in range(100):  # Run for 100 generations
            # Evaluate the fitness of each individual in the population
            fitnesses = [self.evaluate_fitness(individual, logger) for individual in self.population]
            # Select the fittest individuals
            fittest_individuals = np.argsort(fitnesses)[-self.budget:]
            # Select the next generation
            self.population = [self.crossover(individual1, individual2) for individual1, individual2 in zip(self.population[fittest_individuals], fittest_individuals[::2])]
            # Mutate the selected individuals
            for individual in self.population:
                individual, _ = self.mutate(individual)
            # Update the logger
            logger.update(self.population, fitnesses)

        # Return the best solution found
        best_individual = self.population[np.argmax(fitnesses)]
        return best_individual

# Example usage:
def func(x):
    return np.sin(x)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler('log.txt'))

hebbbo = HEBBO(1000, 10)
best_individual = hebbbo.run(func, 1000, 10)
print("Best individual:", best_individual)
print("Best fitness:", hebbbo.evaluate_fitness(best_individual, logger))