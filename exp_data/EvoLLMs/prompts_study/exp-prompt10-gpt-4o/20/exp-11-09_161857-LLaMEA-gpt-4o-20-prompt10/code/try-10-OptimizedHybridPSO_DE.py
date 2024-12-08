import numpy as np

class OptimizedHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 2.0  # Adjusted cognitive component
        self.c2 = 2.0  # Adjusted social component
        self.w = 0.8  # Adjusted inertia weight for better balance
        self.F = 0.4  # More aggressive mutation factor
        self.CR = 0.8  # Adjusted crossover rate
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.adaptive_w_factor = 0.98  # Updated adaptive factor for inertia weight
        self.dynamic_shrink = int(0.7 * self.population_size)  # Dynamic swarm size adjustment

    def __call__(self, func):
        function_evaluations = 0

        while function_evaluations < self.budget:
            current_pop_size = min(self.population_size, self.budget - function_evaluations)
            scores = np.apply_along_axis(func, 1, self.positions[:current_pop_size])
            function_evaluations += current_pop_size
            
            for i in range(current_pop_size):
                if scores[i] < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = scores[i]
                    self.personal_best_positions[i] = self.positions[i]
                if scores[i] < self.global_best_score:
                    self.global_best_score = scores[i]
                    self.global_best_position = self.positions[i]
            
            r1 = np.random.rand(current_pop_size, self.dim)
            r2 = np.random.rand(current_pop_size, self.dim)
            self.velocities[:current_pop_size] = (self.w * self.velocities[:current_pop_size] +
                                                  self.c1 * r1 * (self.personal_best_positions[:current_pop_size] - self.positions[:current_pop_size]) +
                                                  self.c2 * r2 * (self.global_best_position - self.positions[:current_pop_size]))
            self.positions[:current_pop_size] += self.velocities[:current_pop_size]
            self.w *= self.adaptive_w_factor
            
            diversity = np.mean(np.std(self.positions[:current_pop_size], axis=0))
            for i in range(current_pop_size):
                if diversity < 0.15:  # More lenient diversity check
                    indices = list(range(current_pop_size))
                    indices.remove(i)
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    mutant = self.positions[a] + self.F * (self.positions[b] - self.positions[c])
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                    cross_points = np.random.rand(self.dim) < self.CR
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True

                    trial = np.where(cross_points, mutant, self.positions[i])

                    trial_score = func(trial)
                    function_evaluations += 1

                    if trial_score < scores[i]:
                        self.positions[i] = trial
                        scores[i] = trial_score
                        if trial_score < self.personal_best_scores[i]:
                            self.personal_best_scores[i] = trial_score
                            self.personal_best_positions[i] = trial

                    if trial_score < self.global_best_score:
                        self.global_best_score = trial_score
                        self.global_best_position = trial
            
            self.positions[:current_pop_size] = np.clip(self.positions[:current_pop_size], self.lower_bound, self.upper_bound)
            self.population_size = max(self.dynamic_shrink, 10)  # Reduce swarm size over time

        return self.global_best_position