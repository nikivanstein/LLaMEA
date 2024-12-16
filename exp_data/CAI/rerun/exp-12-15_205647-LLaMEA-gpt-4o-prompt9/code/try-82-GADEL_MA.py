import numpy as np

class GADEL_MA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(60, budget // 8)
        self.base_F = 0.5  # Base differential weight
        self.base_CR = 0.9  # Base crossover probability
        self.learning_rate = 0.1
        self.bandit_arms = 5
        self.bandit_counts = np.zeros(self.bandit_arms)
        self.bandit_rewards = np.zeros(self.bandit_arms)

    def __call__(self, func):
        np.random.seed(42)
        
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        best_idx = np.argmin(fitness)
        best = population[best_idx]

        while eval_count < self.budget:
            arm_idx = np.argmax(self.bandit_rewards / (self.bandit_counts + 1e-5))
            self.bandit_counts[arm_idx] += 1
            
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                F_dynamic = self.base_F + (0.1 * arm_idx / (self.bandit_arms - 1))
                mutant = np.clip(x0 + F_dynamic * (x1 - x2), self.lower_bound, self.upper_bound)

                cross_points = np.random.rand(self.dim) < (self.base_CR - 0.1 * arm_idx / (self.bandit_arms - 1))
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                f_trial = func(trial)
                eval_count += 1

                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
                    self.bandit_rewards[arm_idx] += 1
                    if f_trial < fitness[best_idx]:
                        best_idx = i
                        best = trial

            neighborhood_scale = np.clip(0.8 - 0.6 * (eval_count / self.budget), 0.2, 0.8)
            neighborhood = np.clip(best + np.random.normal(0, self.learning_rate * neighborhood_scale, self.dim), self.lower_bound, self.upper_bound)
            f_neighborhood = func(neighborhood)
            eval_count += 1
            if f_neighborhood < fitness[best_idx]:
                best = neighborhood
                fitness[best_idx] = f_neighborhood

        return best, fitness[best_idx]