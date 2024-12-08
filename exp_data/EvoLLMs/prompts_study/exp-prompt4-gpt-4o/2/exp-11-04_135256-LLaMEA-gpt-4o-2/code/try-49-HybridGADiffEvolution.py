import numpy as np

class HybridGADiffEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.elite_ratio = 0.1
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))

    def __call__(self, func):
        num_elites = int(self.elite_ratio * self.population_size)
        evaluations = 0
        scores = np.apply_along_axis(func, 1, self.population)
        evaluations += self.population_size
        prev_best_score = np.min(scores)

        while evaluations < self.budget:
            elite_indices = scores.argsort()[:num_elites]
            elites = self.population[elite_indices]

            next_population = np.empty((self.population_size, self.dim))
            for i in range(self.population_size):
                if i < num_elites:
                    next_population[i] = elites[i]
                else:
                    # Differential Evolution Mutation
                    candidates = np.random.choice(self.population_size, 3, replace=False)
                    a, b, c = self.population[candidates]
                    adaptive_factor = self.mutation_factor * (1.5 if np.min(scores) < prev_best_score else 0.5)
                    adaptive_factor *= 1 + np.exp(-evaluations/self.budget)  # Fine-tune mutation factor
                    mutant = np.clip(a + adaptive_factor * (b - c), self.lower_bound, self.upper_bound)

                    # Adjust crossover probability based on diversity
                    diversity = np.std(self.population)
                    adjusted_crossover_probability = self.crossover_probability * (1 + 0.1 * diversity)

                    # Crossover
                    trial = np.array([mutant[j] if np.random.rand() < adjusted_crossover_probability else self.population[i][j] for j in range(self.dim)])
                    
                    # Selection
                    trial_score = func(trial)
                    evaluations += 1
                    if trial_score < scores[i]:
                        next_population[i] = trial
                        scores[i] = trial_score
                    else:
                        next_population[i] = self.population[i]

                if evaluations >= self.budget:
                    break

            prev_best_score = np.min(scores)
            self.population = next_population

        best_idx = scores.argmin()
        return self.population[best_idx], scores[best_idx]