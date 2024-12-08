import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.strategy = 'rand/1/bin'
        
    def __call__(self, func):
        np.random.seed(42)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.asarray([func(ind) for ind in population])
        evaluations = self.population_size
        success_rate = np.zeros(self.population_size)

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)

                # Adaptive crossover probability based on individual success rates
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
                else:
                    success_rate[i] = 0.9 * success_rate[i]  # Decay success rate gradually

        best_idx = np.argmin(fitness)
        return population[best_idx]