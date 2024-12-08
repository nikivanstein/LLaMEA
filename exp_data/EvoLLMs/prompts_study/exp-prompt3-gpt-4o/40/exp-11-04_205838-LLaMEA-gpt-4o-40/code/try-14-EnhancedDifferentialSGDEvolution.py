import numpy as np

class EnhancedDifferentialSGDEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(20, dim * 5)
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9
        self.partition_count = 3  # Divide dimensions into partitions

    def __call__(self, func):
        eval_count = 0
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        scores = np.array([func(ind) for ind in population])
        eval_count += self.population_size

        best_index = np.argmin(scores)
        best_position = population[best_index]
        best_score = scores[best_index]

        while eval_count < self.budget:
            for i in range(self.population_size):
                indices = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                a, b, c = population[indices]
                
                mutant_vector = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < self.crossover_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial_vector = np.where(cross_points, mutant_vector, population[i])
                trial_score = func(trial_vector)
                eval_count += 1

                if trial_score < scores[i]:
                    population[i] = trial_vector
                    scores[i] = trial_score
                    if trial_score < best_score:
                        best_score = trial_score
                        best_position = trial_vector

            if eval_count < self.budget - self.dim:
                partitions = np.array_split(np.arange(self.dim), self.partition_count)
                for i in range(len(population)):
                    for partition in partitions:
                        grad = np.zeros(self.dim)
                        for d in partition:
                            x_plus = np.copy(population[i])
                            x_plus[d] += 1e-5
                            x_minus = np.copy(population[i])
                            x_minus[d] -= 1e-5

                            grad[d] = (func(x_plus) - func(x_minus)) / (2 * 1e-5)
                            eval_count += 2

                        population[i] -= 0.01 * grad
                        population[i] = np.clip(population[i], self.lower_bound, self.upper_bound)

        return best_position