import numpy as np

class AdvancedHybridPSO_DE_LocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 60  # Increased population for better exploration
        self.c1 = 1.8  # Adjusted cognitive coefficient for balanced exploration
        self.c2 = 0.9  # Adjusted social coefficient
        self.w = 0.5  # Adjusted inertia weight for better convergence dynamics
        self.F_base = 0.5  # Slightly increased mutation factor
        self.CR = 0.9  # Slightly increased crossover rate
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.velocities = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))
        self.positions = self.lower_bound + (self.upper_bound - self.lower_bound) * np.random.rand(self.population_size, self.dim) * np.cos(np.arange(self.population_size)[:, None])
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.adaptive_w_factor = 0.98  # Slightly adjusted adaptive inertia weight
        self.local_search_intensity = 0.1  # Local search intensity parameter

    def __call__(self, func):
        function_evaluations = 0
        while function_evaluations < self.budget:
            scores = np.apply_along_axis(func, 1, self.positions)
            function_evaluations += self.population_size
            
            for i in range(self.population_size):
                if scores[i] < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = scores[i]
                    self.personal_best_positions[i] = self.positions[i]
                if scores[i] < self.global_best_score:
                    self.global_best_score = scores[i]
                    self.global_best_position = self.positions[i]
            
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            self.velocities = (self.w * self.velocities +
                               self.c1 * r1 * (self.personal_best_positions - self.positions) +
                               self.c2 * r2 * (self.global_best_position - self.positions))
            self.positions = self.positions + self.velocities
            self.w *= self.adaptive_w_factor
            
            diversity = np.mean(np.std(self.positions, axis=0))
            for i in range(self.population_size):
                if diversity < 0.25:
                    indices = list(range(self.population_size))
                    indices.remove(i)
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    F_adaptive = self.F_base + 0.3 * (1 - (scores[i] / (self.global_best_score + 1e-9)))
                    mutant = self.positions[a] + F_adaptive * (self.positions[b] - self.positions[c])
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
                        
            for i in range(self.population_size):
                local_search_vector = np.random.uniform(-self.local_search_intensity, self.local_search_intensity, self.dim)
                candidate_position = self.positions[i] + local_search_vector
                candidate_position = np.clip(candidate_position, self.lower_bound, self.upper_bound)
                candidate_score = func(candidate_position)
                function_evaluations += 1
                if candidate_score < scores[i]:
                    scores[i] = candidate_score
                    self.positions[i] = candidate_position
                    if candidate_score < self.personal_best_scores[i]:
                        self.personal_best_scores[i] = candidate_score
                        self.personal_best_positions[i] = candidate_position
                    if candidate_score < self.global_best_score:
                        self.global_best_score = candidate_score
                        self.global_best_position = candidate_position

            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

        return self.global_best_position