import numpy as np

class AdaptiveGeneticAlgorithm:
    def __init__(self, budget, dim, mutation_rate=0.01, crossover_rate=0.5):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = None
        self.fitness_history = []
        self.best_individual = None
        self.best_fitness = float('-inf')

    def __call__(self, func):
        if not self.population:
            self.population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(100)]
        
        while len(self.population) > 0 and self.budget > 0:
            # Select parents using tournament selection
            parents = self.select_parents(self.population)
            
            # Crossover to create offspring
            offspring = self.crossover(parents)
            
            # Mutate offspring with probability
            mutated_offspring = self.mutate(offspring, self.mutation_rate)
            
            # Evaluate fitness of offspring
            fitnesses = self.evaluate_fitness(offspring, func)
            
            # Select best individual
            if len(self.fitness_history) < self.budget:
                self.best_individual = mutated_offspring[np.argmax(fitnesses)]
                self.best_fitness = fitnesses[np.argmax(fitnesses)]
            else:
                self.best_individual = self.best_individual if fitnesses[np.argmax(fitnesses)] > self.best_fitness else mutated_offspring[np.argmax(fitnesses)]
                self.best_fitness = fitnesses[np.argmax(fitnesses)]
            
            # Replace worst individual
            self.population[np.argmax(fitnesses)] = mutated_offspring[np.argmax(fitnesses)]
            self.fitness_history.append(fitnesses[np.argmax(fitnesses)])
            self.budget -= 1
            
            # Stop if all individuals have been evaluated
            if len(self.fitness_history) == self.budget:
                break
        
        # Return best individual
        return self.best_individual

    def select_parents(self, population):
        return np.random.choice(population, size=100, replace=False)

    def crossover(self, parents):
        children = []
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i+1]
            child = (parent1 + parent2) / 2
            children.append(child)
        return children

    def mutate(self, offspring, mutation_rate):
        return offspring * (1 + mutation_rate)

    def evaluate_fitness(self, offspring, func):
        fitnesses = []
        for individual in offspring:
            func(individual)
            fitnesses.append(func(individual))
        return fitnesses

# Example usage:
ga = AdaptiveGeneticAlgorithm(budget=100, dim=10)
ga(func=lambda x: x**2)

# Print initial population and best individual
print("Initial population:")
for individual in ga.population:
    print(individual)
print("\nBest individual:", ga.best_individual)
print("Best fitness:", ga.best_fitness)