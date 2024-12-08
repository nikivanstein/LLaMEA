import numpy as np

class RefinedHybridDifferentialEvolutionWithMultiLevelStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.scaling_factor = 0.8
        self.crossover_rate = 0.9
        self.multi_level_population = True  # New multi-level population strategy

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size
        
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        
        while num_evaluations < self.budget:
            if self.multi_level_population:
                levels = 2 + int(3 * (1 - num_evaluations/self.budget))
                level_size = len(population) // levels
                effective_pop = [population[i*level_size:(i+1)*level_size] for i in range(levels)]
                effective_fit = [fitness[i*level_size:(i+1)*level_size] for i in range(levels)]
            else:
                effective_pop, effective_fit = [population], [fitness]

            trial_population = np.empty_like(population)
            for pop, fit in zip(effective_pop, effective_fit):
                for i in range(len(pop)):
                    indices = np.random.choice(len(pop), 3, replace=False)
                    x0, x1, x2 = pop[indices]
                    mutant_vector = x0 + self.scaling_factor * (x1 - x2)
                    mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                    cross_points = np.random.rand(self.dim) < self.crossover_rate
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True

                    trial_vector = np.where(cross_points, mutant_vector, pop[i])
                    trial_population[i] = trial_vector

            trial_fitness = np.array([func(ind) for ind in trial_population])
            num_evaluations += len(trial_population)

            improvement_mask = trial_fitness < fitness
            population[improvement_mask] = trial_population[improvement_mask]
            fitness[improvement_mask] = trial_fitness[improvement_mask]

            best_idx = np.argmin(fitness)
            if fitness[best_idx] < func(best_individual):
                best_individual = population[best_idx]

            if num_evaluations < self.budget:
                if np.random.rand() < 0.5:
                    perturbation = self.stochastic_levy_perturbation() * (1 + np.abs(func(best_individual)) / np.min(fitness))
                    candidate = best_individual + perturbation * (population[best_idx] - np.mean(population, axis=0))
                    candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                    candidate_fitness = func(candidate)
                    num_evaluations += 1
                    if candidate_fitness < func(best_individual):
                        best_individual = candidate

        return best_individual

    def stochastic_levy_perturbation(self, dim_beta=1.5):
        beta = dim_beta
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size=(self.dim,))
        v = np.random.normal(0, 1, size=(self.dim,))
        step = u / np.abs(v)**(1 / beta)
        return step

# Usage example:
# optimizer = RefinedHybridDifferentialEvolutionWithMultiLevelStrategy(budget=10000, dim=10)
# best_solution = optimizer(my_black_box_function)