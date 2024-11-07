import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 10 * dim
        self.mutation_factor = 0.5 + np.random.rand(self.initial_population_size) * 0.5
        self.crossover_probability = 0.9

    def __call__(self, func):
        np.random.seed(42)
        population_size = self.initial_population_size
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.asarray([func(ind) for ind in population])
        evaluations = population_size
        success_rate = np.zeros(population_size)

        def crowding_distance(individual):
            return np.sum(np.square(individual))

        while evaluations < self.budget:
            for i in range(population_size):
                if evaluations >= self.budget:
                    break

                # Adjust population size dynamically based on progress
                if evaluations % (self.budget // 10) == 0:
                    population_size = max(4, int(population_size * 0.9))
                    population = population[:population_size]
                    fitness = fitness[:population_size]
                    success_rate = success_rate[:population_size]

                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                F = self.mutation_factor[np.random.randint(population_size)]
                mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)

                adaptive_crossover = min(1.0, self.crossover_probability + 0.1 * (1 - success_rate[i]))
                cross_points = np.random.rand(self.dim) < (adaptive_crossover * np.random.rand())

                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                f_trial = func(trial)
                evaluations += 1

                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    success_rate[i] = 1
                else:
                    success_rate[i] = 0.9 * success_rate[i]

                crowd_distances = np.asarray([crowding_distance(ind) for ind in population])
                least_crowded_idx = np.argmax(crowd_distances)
                if fitness[least_crowded_idx] > f_trial:
                    population[least_crowded_idx] = trial
                    fitness[least_crowded_idx] = f_trial

                tournament_size = 3
                tournament_idxs = np.random.choice(population_size, tournament_size, replace=False)
                best_t_idx = tournament_idxs[np.argmin(fitness[tournament_idxs])]
                if fitness[best_t_idx] > f_trial:
                    population[best_t_idx] = trial
                    fitness[best_t_idx] = f_trial

        best_idx = np.argmin(fitness)
        return population[best_idx]