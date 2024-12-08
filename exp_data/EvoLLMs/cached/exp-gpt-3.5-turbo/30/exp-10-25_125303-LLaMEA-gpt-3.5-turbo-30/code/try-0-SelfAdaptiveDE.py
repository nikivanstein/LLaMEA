import numpy as np

class SelfAdaptiveDE:
    def __init__(self, budget, dim, population_size=10, f=0.5, cr=0.5):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f = f
        self.cr = cr

    def __call__(self, func):
        def create_population():
            return np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        def mutate(target, pop):
            candidates = np.random.choice(pop, size=(3, self.dim), replace=False)
            mutant = target + self.f * (candidates[0] - candidates[1]) + self.f * (candidates[2] - candidates[3])
            return np.clip(mutant, -5.0, 5.0)

        def crossover(target, mutant):
            trial = np.copy(target)
            idx = np.random.rand(self.dim) < self.cr
            trial[idx] = mutant[idx]
            return trial

        population = create_population()
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        remaining_budget = self.budget - self.population_size

        while remaining_budget > 0:
            new_population = []
            for idx, target in enumerate(population):
                mutant = mutate(target, population)
                trial = crossover(target, mutant)
                fitness_trial = func(trial)
                if fitness_trial < fitness[idx]:
                    population[idx] = trial
                    fitness[idx] = fitness_trial
                    if fitness_trial < fitness[best_idx]:
                        best_idx = idx
                        best_solution = trial
                remaining_budget -= 1
                if remaining_budget <= 0:
                    break

        return best_solution

# Example usage:
# optimizer = SelfAdaptiveDE(budget=1000, dim=10)
# result = optimizer(lambda x: np.sum(x**2))  # Optimize the sphere function