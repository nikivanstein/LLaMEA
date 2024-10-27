import numpy as np

class EvolutionarySpaceShifting:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5
        self.initial_population = self.generate_initial_population()

    def generate_initial_population(self):
        return [np.random.uniform(self.lower_bound, self.upper_bound, self.dim) for _ in range(self.population_size)]

    def fitness(self, x):
        return np.mean(np.abs(np.random.normal(0, 1, x.shape) * (x - self.lower_bound) + self.upper_bound))

    def evolve(self, func):
        for _ in range(self.budget):
            # Select parents
            parents = np.array([np.random.choice(self.initial_population, 2, replace=False) for _ in range(self.population_size)])

            # Calculate fitness
            fitnesses = np.array([self.fitness(parent) for parent in parents])

            # Select fittest
            fittest_indices = np.argsort(fitnesses)[:int(self.population_size * 0.2)]
            fittest_parents = parents[fittest_indices]

            # Crossover
            offspring = []
            for _ in range(self.population_size):
                parent1, parent2 = np.random.choice(fittest_parents, 2, replace=False)
                child = (parent1 + parent2) / 2
                if np.random.rand() < self.crossover_rate:
                    child = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                offspring.append(child)

            # Mutation
            mutated_offspring = []
            for child in offspring:
                if np.random.rand() < self.mutation_rate:
                    mutated_child = child + np.random.uniform(-0.1, 0.1, self.dim)
                    mutated_child = np.clip(mutated_child, self.lower_bound, self.upper_bound)
                mutated_offspring.append(mutated_child)

            # Replace worst individuals
            self.initial_population = mutated_offspring

            # Evaluate fitness
            fitnesses = np.array([self.fitness(x) for x in self.initial_population])
            self.initial_population = self.initial_population[np.argsort(fitnesses)]

    def __call__(self, func):
        self.evolve(func)
        return self.initial_population[np.argmin([self.fitness(x) for x in self.initial_population])]

# Usage
if __name__ == "__main__":
    budget = 100
    dim = 10
    algorithm = EvolutionarySpaceShifting(budget, dim)
    best_solution = algorithm(np)
    print("Best solution:", best_solution)