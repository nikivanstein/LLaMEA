import numpy as np

class Hybrid_PSO_GA_Optimization:
    def __init__(self, budget, dim, population_size=30, c1=2.0, c2=2.0, mutation_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.c1 = c1
        self.c2 = c2
        self.mutation_rate = mutation_rate

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        def update_velocity(position, velocity, pbest, gbest):
            new_velocity = velocity + self.c1 * np.random.rand() * (pbest - position) + self.c2 * np.random.rand() * (gbest - position)
            return new_velocity

        def mutate(individual):
            for i in range(len(individual)):
                if np.random.rand() < self.mutation_rate:
                    individual[i] += np.random.uniform(-1, 1)
            return individual

        population = initialize_population()
        velocities = np.zeros_like(population)
        pbest = population.copy()
        gbest = population[np.argmin([func(ind) for ind in population])]
        remaining_budget = self.budget - self.population_size

        while remaining_budget > 0:
            for idx, individual in enumerate(population):
                if remaining_budget <= 0:
                    break
                velocities[idx] = update_velocity(individual, velocities[idx], pbest[idx], gbest)
                new_individual = individual + velocities[idx]
                new_individual = mutate(new_individual)
                new_fitness = func(new_individual)

                if new_fitness < func(individual):
                    population[idx] = new_individual
                    if new_fitness < func(pbest[idx]):
                        pbest[idx] = new_individual
                        if new_fitness < func(gbest):
                            gbest = new_individual
                remaining_budget -= 1

        return gbest

# Example usage:
# optimizer = Hybrid_PSO_GA_Optimization(budget=1000, dim=10)
# result = optimizer(lambda x: np.sum(x**2))  # Optimize the sphere function