import numpy as np

class AdaptiveReservoirSampling:
    def __init__(self, budget, dim, reservoir_size=100, refinement_threshold=0.2):
        self.budget = budget
        self.dim = dim
        self.reservoir_size = reservoir_size
        self.refinement_threshold = refinement_threshold
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def reservoir_sampling(population, func, bounds, reservoir_size, threshold):
            reservoir = []
            for individual in population:
                fitness = objective(individual)
                if fitness < threshold:
                    reservoir.append(individual)
                if len(reservoir) >= reservoir_size:
                    break
            return reservoir

        def adaptive_refinement(reservoir, bounds):
            if len(reservoir) == 0:
                return bounds
            elif len(reservoir) < self.reservoir_size:
                return bounds
            else:
                # Refine the reservoir based on the fitness values
                fitness_values = np.array([objective(individual) for individual in reservoir])
                sorted_indices = np.argsort(fitness_values)
                sorted_reservoir = reservoir[sorted_indices[:self.reservoir_size]]
                return adaptive_refinement(sorted_reservoir, bounds)

        def evaluate_fitness(reservoir, bounds):
            updated_individuals = []
            for individual in reservoir:
                fitness = objective(individual)
                if fitness < bounds[0] + 1e-6:
                    updated_individuals.append(individual)
                if len(updated_individuals) >= self.budget:
                    break
            return updated_individuals

        def mutate(individual, bounds):
            if np.random.rand() < 0.5:
                individual = np.random.uniform(bounds[0], bounds[1])
            return individual

        reservoir = reservoir_sampling(self.population, objective, bounds, self.reservoir_size, self.refinement_threshold)
        while len(reservoir) < self.budget:
            new_individual = mutate(reservoir[-1], bounds)
            fitness = objective(new_individual)
            if fitness < bounds[0] + 1e-6:
                reservoir.append(new_individual)
            if len(reservoir) >= self.budget:
                break
        return evaluate_fitness(reservoir, bounds)

# Description: Adaptive Reservoir Sampling with Adaptive Bound Refinement
# Code: 