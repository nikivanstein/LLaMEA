import numpy as np

class ImprovedDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.bounds = [-5.0, 5.0]
        self.c1 = 1.4  # Adjusted cognitive parameter
        self.c2 = 1.6  # Adjusted social parameter
        self.w = 0.7  # Modified inertia weight for improved exploration
        self.F = 0.9  # Enhanced DE Mutation factor
        self.CR = 0.6  # Adjusted DE Crossover probability
        self.T = 1.0
        self.T_min = 1e-4
        self.alpha = 0.95  # Cooling rate
        self.diversity_threshold = 0.1  # Diversity threshold for local search

    def dynamic_parameters(self, evaluations):
        # Adjust DE parameters dynamically based on function evaluations and diversity
        self.F = 0.6 + (0.3 - evaluations / self.budget) * np.random.rand()
        self.CR = 0.6 + (0.4 - evaluations / self.budget) * np.random.rand()
        self.T = max(self.T_min, self.T * self.alpha)

    def diversity_measure(self, population):
        return np.mean(np.std(population, axis=0))

    def local_search(self, position, func):
        perturbation = np.random.uniform(-0.1, 0.1, position.shape)
        candidate = np.clip(position + perturbation, self.bounds[0], self.bounds[1])
        return candidate if func(candidate) < func(position) else position

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

            for i in range(self.pop_size):
                if self.diversity_measure(population) < self.diversity_threshold:
                    population[i] = self.local_search(population[i], func)

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