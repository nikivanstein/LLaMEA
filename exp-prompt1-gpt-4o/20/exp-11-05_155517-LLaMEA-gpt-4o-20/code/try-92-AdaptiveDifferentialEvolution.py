import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.mutation_factor = 0.5 + np.random.rand(self.population_size) * 0.5  # Dynamic mutation factor
        self.crossover_probability = 0.9

    def __call__(self, func):
        np.random.seed(42)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.asarray([func(ind) for ind in population])
        evaluations = self.population_size
        success_rate = np.zeros(self.population_size)
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()

        def crowding_distance(individual):
            return np.sum(np.square(individual))

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]

                # Adaptive mutation factor based on fitness diversity
                fitness_range = np.ptp(fitness)
                F = self.mutation_factor[np.random.randint(self.population_size)] * max(0.1, fitness_range / (np.max(fitness) + 1e-9))
                mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)

                adaptive_crossover = min(1.0, self.crossover_probability + 0.1 * (1 - success_rate[i]))
                cross_points = np.random.rand(self.dim) < adaptive_crossover

                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                f_trial = func(trial)
                evaluations += 1

                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    success_rate[i] = 1
                    if f_trial < fitness[best_idx]:
                        best_idx = i
                        best_solution = trial.copy()
                else:
                    success_rate[i] = 0.9 * success_rate[i]

                # Incorporate crowding distance for diversity
                crowd_distances = np.asarray([crowding_distance(ind) for ind in population])
                least_crowded_idx = np.argmax(crowd_distances)
                if fitness[least_crowded_idx] > f_trial:
                    population[least_crowded_idx] = trial
                    fitness[least_crowded_idx] = f_trial

                # Tournament selection to enhance diversity
                tournament_size = 3
                tournament_idxs = np.random.choice(self.population_size, tournament_size, replace=False)
                best_t_idx = tournament_idxs[np.argmin(fitness[tournament_idxs])]
                if fitness[best_t_idx] > f_trial:
                    population[best_t_idx] = trial
                    fitness[best_t_idx] = f_trial

        return best_solution