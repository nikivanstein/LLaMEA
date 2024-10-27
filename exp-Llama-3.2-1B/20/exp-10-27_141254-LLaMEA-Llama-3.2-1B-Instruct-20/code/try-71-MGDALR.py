import numpy as np

class MGDALR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.learning_rate = 0.01
        self.explore_count = 0
        self.max_explore_count = budget
        self.explore_history = []

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
            
            # Store the exploration history
            self.explore_history.append((x, y))
        
        return x

class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.learning_rate = 0.01
        self.explore_count = 0
        self.max_explore_count = budget
        self.population_size = 100
        self.population = self.initialize_population()
        self.fitness_scores = []
    
    def initialize_population(self):
        return [np.array([-5.0] * self.dim) for _ in range(self.population_size)]
    
    def __call__(self, func):
        # Create a new population by evolving the existing one
        new_population = self.population.copy()
        
        for _ in range(self.budget):
            # Select parents using tournament selection
            parents = self.select_parents(new_population)
            
            # Crossover (reproduce) offspring using single-point crossover
            offspring = self.crossover(parents)
            
            # Mutate offspring using single-point mutation
            offspring = self.mutate(offspring)
            
            # Replace the old population with the new one
            new_population = self.update_population(new_population, func, offspring)
        
        return new_population
    
    def select_parents(self, population):
        # Select parents using tournament selection
        winners = []
        for _ in range(self.population_size):
            winner = np.random.choice(population, p=[1 - self.explore_rate / self.population_size, self.explore_rate])
            winners.append(winner)
        
        return winners
    
    def crossover(self, parents):
        # Crossover (reproduce) offspring using single-point crossover
        offspring = []
        for i in range(self.population_size):
            parent1 = parents[i]
            parent2 = parents[(i + 1) % self.population_size]
            child = np.copy(parent1)
            for j in range(self.dim):
                if np.random.rand() < 0.5:
                    child[j] = parent2[j]
            offspring.append(child)
        
        return offspring
    
    def mutate(self, offspring):
        # Mutate offspring using single-point mutation
        for i in range(self.population_size):
            if np.random.rand() < 0.1:
                offspring[i] = np.random.uniform(-1, 1, self.dim)
        
        return offspring
    
    def update_population(self, population, func, offspring):
        # Replace the old population with the new one
        new_population = []
        for _ in range(self.population_size):
            new_individual = func(offspring[_])
            new_population.append(new_individual)
        
        return new_population

# One-line description with the main idea:
# Evolutionary Algorithm for Black Box Optimization using Genetic Algorithm