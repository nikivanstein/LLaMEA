import numpy as np

class GDSCMetaheuristic:
    def __init__(self, budget, dim, population_size=100, mutation_rate=0.01):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = None
        self.fitness_scores = {}

    def __call__(self, func):
        # Evaluate the function 1 time
        self.func_values[func.__name__] = func()
        
        # Initialize the population randomly
        if self.population is None:
            self.population = np.random.uniform(self.search_space[0], self.search_space[1], (self.population_size, self.dim))
        
        # Evaluate the fitness of each individual in the population
        fitness_scores = np.array([self.evaluate_fitness(individual) for individual in self.population])
        
        # Select the fittest individuals
        self.population = self.select_fittest(population_scores=fitness_scores, num_to_select=self.population_size)
        
        # Update the function values for the next iteration
        for _ in range(self.budget):
            new_population = self.update_population(population=fitness_scores, mutation_rate=self.mutation_rate)
            self.population = new_population
        
        # Reassign each individual to the closest cluster center
        self.population = self.reassign_cluster_centers(population=fitness_scores)
        
        # Evaluate the function 1 time
        self.func_values[func.__name__] = func()

    def select_fittest(self, population_scores, num_to_select):
        # Select the fittest individuals based on their fitness scores
        scores = np.array(population_scores)
        indices = np.argsort(scores)[::-1][:num_to_select]
        return [self.population[i] for i in indices]

    def update_population(self, population_scores, mutation_rate):
        # Update the population with new individuals
        new_population = np.random.uniform(self.search_space[0], self.search_space[1], (self.population_size, self.dim))
        for i in range(self.population_size):
            new_individual = self.evaluate_fitness(new_population[i])
            new_individual = self.fitness_to_individual(new_individual, self.func_values)
            new_individual = self.mutate(individual=new_individual, mutation_rate=mutation_rate)
            new_population[i] = new_individual
        
        return new_population

    def reassign_cluster_centers(self, population_scores):
        # Reassign each individual to the closest cluster center
        cluster_centers = np.array([self.cluster_centers])
        for individual in population_scores:
            dist = np.linalg.norm(individual - cluster_centers, axis=1)
            cluster_centers = np.argmin(dist, axis=0)
        
        return cluster_centers

    def fitness_to_individual(self, fitness, func):
        # Convert the fitness score to an individual
        individual = np.zeros(self.dim)
        for i in range(self.dim):
            individual[i] = fitness[i]
        return individual

    def mutate(self, individual, mutation_rate):
        # Mutate the individual with a small probability
        if np.random.rand() < mutation_rate:
            return individual + np.random.uniform(-1, 1, self.dim)
        return individual

    def evaluate_fitness(self, individual):
        # Evaluate the fitness of the individual
        return func(individual)

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 