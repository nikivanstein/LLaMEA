import numpy as np

class EnhancedAdaptiveMultiStrategyDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.bounds = [-5.0, 5.0]
        self.c1 = 1.7  # Adjusted cognitive parameter for better exploration
        self.c2 = 1.3  # Adjusted social parameter for improved convergence
        self.w_max = 0.9  # Dynamic inertia weight range start
        self.w_min = 0.4  # Dynamic inertia weight range end
        self.F = 0.8
        self.CR = 0.7
        self.T_min = 1e-3
        self.alpha = 0.95  # Faster cooling rate

    def dynamic_parameters(self, evaluations):
        # Adjust DE parameters dynamically based on function evaluations
        self.F = 0.5 + (0.5 - evaluations / self.budget) * np.random.rand()
        self.CR = 0.5 + (0.5 - evaluations / self.budget) * np.random.rand()
        self.w = self.w_max - ((self.w_max - self.w_min) * (evaluations / self.budget))

    def stochastic_local_search(self, candidate, score, func):
        # Introduce small perturbations for local search
        perturbation = np.random.uniform(-0.1, 0.1, self.dim)
        new_candidate = np.clip(candidate + perturbation, self.bounds[0], self.bounds[1])
        new_score = func(new_candidate)
        return (new_candidate, new_score) if new_score < score else (candidate, score)

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
                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial
                    population[i] = trial

                # Apply stochastic local search
                population[i], personal_best_scores[i] = self.stochastic_local_search(population[i], personal_best_scores[i], func)

                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.w * velocities[i] + self.c1 * r1 * (personal_best_positions[i] - population[i]) + self.c2 * r2 * (global_best_position - population[i]))
                population[i] = population[i] + velocities[i]
                population[i] = np.clip(population[i], self.bounds[0], self.bounds[1])

                trial_score = func(population[i])
                evaluations += 1

                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = population[i]

                if trial_score < global_best_score:
                    global_best_score = trial_score
                    global_best_position = population[i]

                if evaluations >= self.budget:
                    break

        return global_best_position, global_best_score