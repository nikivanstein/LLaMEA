import numpy as np

class PSO_DE_Hybrid:
    def __init__(self, budget, dim, population_size=30, w=0.5, c1=1.49445, c2=1.49445, f=0.5, cr=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.f = f
        self.cr = cr

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        def pso_de_step(population, velocities, best_individual):
            new_population = []
            for idx, target in enumerate(population):
                # PSO update
                velocities[idx] = self.w * velocities[idx] + self.c1 * np.random.rand(self.dim) * (best_individual - target) + self.c2 * np.random.rand(self.dim) * (population[np.random.randint(0, self.population_size)] - target)
                target += velocities[idx]

                # DE update
                mutant = target + self.f * (population[np.random.randint(0, self.population_size)] - population[np.random.randint(0, self.population_size)])
                crossover_points = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover_points, mutant, target)

                new_population.append(trial)
            return np.array(new_population), velocities

        population = initialize_population()
        velocities = np.zeros((self.population_size, self.dim))
        best_individual = population[np.argmin([func(ind) for ind in population])]
        remaining_budget = self.budget - self.population_size

        while remaining_budget > 0:
            new_population, velocities = pso_de_step(population, velocities, best_individual)
            for idx, individual in enumerate(new_population):
                if remaining_budget <= 0:
                    break
                new_fitness = func(individual)
                if new_fitness < func(population[idx]):
                    population[idx] = individual
                    if new_fitness < func(best_individual):
                        best_individual = individual
                remaining_budget -= 1

        return best_individual

# Example usage:
# optimizer = PSO_DE_Hybrid(budget=1000, dim=10)
# result = optimizer(lambda x: np.sum(x**2))  # Optimize the sphere function