import numpy as np

class AdaptiveMultiPopDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 40  # Increased population size for diversity
        self.sub_pop_size = 20  # Sub-population size
        self.bounds = [-5.0, 5.0]
        self.c1 = 1.4  # Adjusted cognitive parameter
        self.c2 = 1.4  # Adjusted social parameter
        self.w = 0.7  # Increased inertia weight
        self.F = 0.9  # Adjusted DE Mutation factor
        self.CR = 0.6  # Adjusted DE Crossover probability
        self.T = 1.0
        self.T_min = 1e-3
        self.alpha = 0.85  # Adjusted cooling rate

    def dynamic_parameters(self, evaluations):
        self.F = 0.4 + (0.4 - evaluations / self.budget) * np.random.rand()
        self.CR = 0.5 + (0.3 - evaluations / self.budget) * np.random.rand()
        self.T = max(self.T_min, self.T * self.alpha)

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

            # Split population into sub-populations and process them
            for k in range(0, self.pop_size, self.sub_pop_size):
                sub_population = population[k:k+self.sub_pop_size]
                sub_velocities = velocities[k:k+self.sub_pop_size]
                sub_personal_best_positions = personal_best_positions[k:k+self.sub_pop_size]
                sub_personal_best_scores = personal_best_scores[k:k+self.sub_pop_size]

                for i in range(self.sub_pop_size):
                    indices = [index for index in range(self.sub_pop_size) if index != i]
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    mutant = sub_population[a] + self.F * (sub_population[b] - sub_population[c])
                    mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                    trial = np.copy(sub_population[i])
                    for j in range(self.dim):
                        if np.random.rand() < self.CR:
                            trial[j] = mutant[j]

                    trial_score = func(trial)
                    evaluations += 1
                    if trial_score < sub_personal_best_scores[i] or np.exp((sub_personal_best_scores[i] - trial_score) / self.T) > np.random.rand():
                        sub_personal_best_scores[i] = trial_score
                        sub_personal_best_positions[i] = trial
                        sub_population[i] = trial

                    r1, r2 = np.random.rand(), np.random.rand()
                    sub_velocities[i] = (self.w * sub_velocities[i] + self.c1 * r1 * (sub_personal_best_positions[i] - sub_population[i]) + self.c2 * r2 * (global_best_position - sub_population[i]))
                    trial = sub_population[i] + sub_velocities[i]
                    trial = np.clip(trial, self.bounds[0], self.bounds[1])

                    trial_score = func(trial)
                    evaluations += 1

                    if trial_score < sub_personal_best_scores[i]:
                        sub_personal_best_scores[i] = trial_score
                        sub_personal_best_positions[i] = trial
                        sub_population[i] = trial

                    if trial_score < global_best_score:
                        global_best_score = trial_score
                        global_best_position = trial

                    if evaluations >= self.budget:
                        break

            # Integrate updated sub-populations back to the main population
            population[k:k+self.sub_pop_size] = sub_population
            velocities[k:k+self.sub_pop_size] = sub_velocities
            personal_best_positions[k:k+self.sub_pop_size] = sub_personal_best_positions
            personal_best_scores[k:k+self.sub_pop_size] = sub_personal_best_scores

        return global_best_position, global_best_score