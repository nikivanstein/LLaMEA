import numpy as np

class EnhancedDynamicControlEMODE_Tournament:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        cr = 0.9
        f = 0.8
        scale = 0.1
        bounds = (-5.0, 5.0)

        def clip(x):
            return np.clip(x, bounds[0], bounds[1])

        def initialize_population(population_size):
            return np.random.uniform(bounds[0], bounds[1], size=(population_size, self.dim))

        def tournament_selection(population, fitness_values, size):
            selected_indices = []
            for _ in range(size):
                contestants = np.random.choice(len(population), 2, replace=False)
                winner = contestants[np.argmin([fitness_values[contestants[0]], fitness_values[contestants[1]]])]
                selected_indices.append(winner)
            return population[selected_indices]

        def crowding_distance(population, fitness_values):
            distances = np.zeros(len(population))
            sorted_indices = np.argsort(fitness_values)
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf
            for i in range(1, len(sorted_indices) - 1):
                distances[sorted_indices[i]] += fitness_values[sorted_indices[i+1]] - fitness_values[sorted_indices[i-1]]
            return distances

        def selection(population, fitness_values, size):
            distances = crowding_distance(population, fitness_values)
            selected_indices = distances.argsort()[-size:]
            return population[selected_indices]

        def differential_evolution(population, population_size):
            new_population = np.zeros_like(population)
            fitness_values = [func(ind) for ind in population]
            selected_population = tournament_selection(population, fitness_values, population_size)
            variance = np.var(fitness_values)
            for i in range(population_size):
                target = selected_population[i]
                candidates = np.delete(selected_population, i, axis=0)
                r1, r2, r3 = candidates[np.random.choice(range(len(candidates)), 3, replace=False)]
                noise = np.random.standard_cauchy(self.dim) * scale * (1 + variance)
                mutant = clip(r1 + f * (r2 - r3) + noise)
                crossover_points = np.random.rand(self.dim) < cr
                offspring = np.where(crossover_points, mutant, target)
                new_population[i] = clip(offspring)
            return new_population

        evaluations = 0
        population_size = 10
        population = initialize_population(population_size)
        while evaluations < self.budget:
            f = max(0.4, f - 0.0005)
            cr = min(0.95, cr + 0.0005)

            offspring = differential_evolution(population, population_size)
            for ind in offspring:
                fitness = func(ind)
                evaluations += 1
                if evaluations >= self.budget:
                    break
            population = np.vstack((population, offspring))
            population_size = min(50, int(population_size * 1.1))
        best_solution = population[np.argmin([func(ind) for ind in population])]
        return best_solution