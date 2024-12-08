import numpy as np
from scipy.optimize import minimize

class NicheEnsemble:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.niche_size = 5
        self.population_size = 100
        self.niche_count = 0
        self.niche_centers = []
        self.fitness_history = []
        self.ensemble_size = 10

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

            # Create ensemble
            ensemble = np.zeros((self.ensemble_size, self.dim))
            for i in range(self.ensemble_size):
                ensemble[i] = np.random.choice(self.niche_centers)

            # Evaluate ensemble
            ensemble_fitness = np.zeros(self.ensemble_size)
            for i in range(self.ensemble_size):
                ensemble_fitness[i] = func(ensemble[i])

            # Select best individual
            best_individual = np.argmax(ensemble_fitness)
            best_individual = ensemble[best_individual]

            # Store new fitness history
            self.fitness_history.append(ensemble_fitness[best_individual])
            self.fitness_history.append(func(best_individual))

            # Update population
            population = np.array([best_individual])
            for _ in range(self.population_size - 1):
                new_individual = np.random.uniform(-5.0, 5.0, size=(self.dim,))
                new_individual = self.niche_centers[np.random.randint(0, self.niche_count)]
                population = np.vstack((population, new_individual))

# Example usage
def func(x):
    return np.sum(x**2)

niche_ensemble = NicheEnsemble(budget=100, dim=10)
niche_ensemble(func)