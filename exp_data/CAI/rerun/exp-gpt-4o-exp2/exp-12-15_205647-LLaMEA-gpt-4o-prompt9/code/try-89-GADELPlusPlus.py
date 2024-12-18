import numpy as np

class GADELPlusPlus:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(80, budget // 6)
        self.F = 0.6  # Initial differential weight
        self.CR = 0.85  # Initial crossover probability
        self.learning_rate = 0.15

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
                F_dynamic = self.F * (1 + np.random.normal(0, 0.1))  # More robust perturbation
                mutant = np.clip(x0 + F_dynamic * (x1 - x2), self.lower_bound, self.upper_bound)

                CR_dynamic = self.CR + 0.15 * np.sin(eval_count / self.budget * 2 * np.pi)  # Adaptive crossover
                cross_points = np.random.rand(self.dim) < CR_dynamic
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
                        
                # Introduce stochastic elitism
                if np.random.rand() < 0.1 and f_trial < fitness[best_idx]:
                    population[np.random.choice(self.population_size)] = trial

            if eval_count % (self.population_size * 3) == 0:
                self.F = 0.5 + 0.3 * (eval_count / self.budget)
                self.CR = 0.88 - 0.4 * (eval_count / self.budget)

            neighborhood_scale = np.clip(0.9 - 0.7 * (eval_count / self.budget), 0.25, 0.9)
            neighborhood = np.clip(best + np.random.normal(0, self.learning_rate * neighborhood_scale, self.dim), self.lower_bound, self.upper_bound)
            f_neighborhood = func(neighborhood)
            eval_count += 1
            if f_neighborhood < fitness[best_idx]:
                best = neighborhood
                fitness[best_idx] = f_neighborhood

        return best, fitness[best_idx]