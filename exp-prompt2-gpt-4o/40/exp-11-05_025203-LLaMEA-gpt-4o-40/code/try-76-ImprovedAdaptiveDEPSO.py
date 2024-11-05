import numpy as np

class ImprovedAdaptiveDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.bounds = [-5.0, 5.0]
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.6
        self.F = 0.8
        self.CR = 0.7
        self.T = 1.0
        self.T_min = 1e-3
        self.alpha = 0.9
        self.diversity_threshold = 0.2
        self.min_pop_size = 10

    def dynamic_parameters(self, evaluations):
        self.F = 0.5 + (0.3 - evaluations / self.budget) * np.random.rand()
        self.CR = 0.5 + (0.4 - evaluations / self.budget) * np.random.rand()
        self.T = max(self.T_min, self.T * self.alpha)

    def calculate_diversity(self, population):
        mean_position = np.mean(population, axis=0)
        diversity = np.mean(np.linalg.norm(population - mean_position, axis=1))
        return diversity

    def resize_population(self, diversity):
        if diversity < self.diversity_threshold and self.pop_size > self.min_pop_size:
            self.pop_size = max(self.min_pop_size, int(self.pop_size * 0.9))
        else:
            self.pop_size = min(50, int(self.pop_size * 1.1))

    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.full(self.pop_size, float('inf'))
        global_best_position = None
        global_best_score = float('inf')

        evaluations = 0
        
        while evaluations < self.budget:
            self.dynamic_parameters(evaluations)
            diversity = self.calculate_diversity(population)
            self.resize_population(diversity)

            for i in range(self.pop_size):
                score = func(population[i])
                evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = population[i]
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = population[i]
                if evaluations >= self.budget:
                    break

            for i in range(self.pop_size):
                indices = [index for index in range(self.pop_size) if index != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c])
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                trial = np.copy(population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.CR:
                        trial[j] = mutant[j]

                trial_score = func(trial)
                evaluations += 1
                if trial_score < personal_best_scores[i] or np.exp((personal_best_scores[i] - trial_score) / self.T) > np.random.rand():
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial
                    population[i] = trial

                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.w * velocities[i] + self.c1 * r1 * (personal_best_positions[i] - population[i]) + self.c2 * r2 * (global_best_position - population[i]))
                trial = population[i] + velocities[i]
                trial = np.clip(trial, self.bounds[0], self.bounds[1])

                trial_score = func(trial)
                evaluations += 1

                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial
                    population[i] = trial

                if trial_score < global_best_score:
                    global_best_score = trial_score
                    global_best_position = trial

                if evaluations >= self.budget:
                    break

        return global_best_position, global_best_score