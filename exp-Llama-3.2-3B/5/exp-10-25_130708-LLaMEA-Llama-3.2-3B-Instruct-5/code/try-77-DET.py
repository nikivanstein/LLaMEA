import numpy as np
from scipy.optimize import minimize

class DET:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.niche_size = 5
        self.population_size = 100
        self.niche_count = 0
        self.niche_centers = []
        self.fitness_history = []
        self.mutation_prob = 0.05

    def __call__(self, func):
        # Initialize population with random niches
        population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        for i in range(self.population_size):
            self.niche_count += 1
            if self.niche_count > self.niche_size:
                self.niche_centers.append(self.niche_size)
                self.niche_count = 1
            population[i] = self.niche_centers[np.random.randint(0, self.niche_count)]

        # Evaluate population and store fitness history
        fitness_history = []
        for x in population:
            f = func(x)
            fitness_history.append(f)
            self.fitness_history.append(f)

        # Main loop
        for _ in range(self.budget):
            # Select fittest individuals
            fittest_individuals = np.argsort(fitness_history)[-self.population_size:]
            fittest_population = population[fittest_individuals]

            # Calculate niches
            niches = np.array_split(fittest_population, self.niche_count)
            niches = np.array([np.mean(niche, axis=0) for niche in niches])

            # Update niche centers
            self.niche_centers = niches

            # Evaluate new population
            new_population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            for i in range(self.population_size):
                # Randomly decide to mutate
                if np.random.rand() < self.mutation_prob:
                    new_population[i] = self.mutate(new_population[i])
                new_population[i] = self.niche_centers[np.random.randint(0, self.niche_count)]

            # Store new fitness history
            new_fitness_history = []
            for x in new_population:
                f = func(x)
                new_fitness_history.append(f)
                self.fitness_history.append(f)

            # Update population and fitness history
            population = new_population
            fitness_history = new_fitness_history

    def mutate(self, individual):
        # Perform mutation with 5% probability
        if np.random.rand() < self.mutation_prob:
            # Randomly select two dimensions to mutate
            idx1, idx2 = np.random.randint(0, self.dim, size=2)
            # Randomly decide the direction of mutation
            direction = np.random.choice([-1, 1])
            # Perform mutation
            individual[idx1] += direction * np.random.uniform(-1, 1)
            individual[idx2] += direction * np.random.uniform(-1, 1)
        return individual

# Example usage
def func(x):
    return np.sum(x**2)

det = DET(budget=100, dim=10)
det(func)