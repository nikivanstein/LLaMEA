import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(50, budget // 10)
        self.scaling_factor = 0.8
        self.crossover_rate = 0.9
        self.adaptation_rate = 1.0

    def __call__(self, func):
        np.random.seed(42)
        
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        scores = np.array([func(x) for x in population])
        eval_count = self.population_size

        while eval_count < self.budget:
            new_population = np.copy(population)
            for i in range(self.population_size):
                idxs = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[idxs]
                mutant_vector = a + self.scaling_factor * (b - c)
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                crossover = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True

                trial_vector = np.where(crossover, mutant_vector, population[i])
                trial_score = func(trial_vector)
                eval_count += 1

                if trial_score < scores[i]:
                    new_population[i] = trial_vector
                    scores[i] = trial_score

            population = new_population
            
            self.scaling_factor = 0.8 + 0.2 * (eval_count / self.budget)
            self.crossover_rate = 0.9 - 0.4 * (eval_count / self.budget) + np.random.uniform(-0.1, 0.1)

        best_idx = np.argmin(scores)
        return population[best_idx], scores[best_idx]