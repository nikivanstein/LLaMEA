import numpy as np

class DE_GADEL:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(50, budget // 10)
        self.learning_rate = 0.1

    def __call__(self, func):
        np.random.seed(42)

        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        best_idx = np.argmin(fitness)
        best = population[best_idx]

        while eval_count < self.budget:
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                F = 0.5 + 0.1 * np.sin(eval_count / self.budget * np.pi)  # Adaptive mutation factor
                mutant = np.clip(x0 + F * (x1 - x2), self.lower_bound, self.upper_bound)

                CR = 0.9 - 0.4 * (eval_count / self.budget)  # Dynamic crossover probability
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])
                f_trial = func(trial)
                eval_count += 1

                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
                    if f_trial < fitness[best_idx]:
                        best_idx = i
                        best = trial

            if eval_count % self.population_size == 0:
                self.learning_rate = 0.05 + 0.15 * (eval_count / self.budget)

            neighborhood_scale = np.clip(0.7 - 0.5 * (eval_count / self.budget), 0.2, 0.7)
            neighborhood = np.clip(best + np.random.normal(0, self.learning_rate * neighborhood_scale, self.dim), self.lower_bound, self.upper_bound)
            f_neighborhood = func(neighborhood)
            eval_count += 1
            if f_neighborhood < fitness[best_idx]:
                best = neighborhood
                fitness[best_idx] = f_neighborhood

        return best, fitness[best_idx]