import numpy as np

class RefinedHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 60  # Slightly increased population size for better exploration
        self.c1 = 1.5  # Adjusted cognitive coefficient for balanced exploration and exploitation
        self.c2 = 1.2  # Adjusted social coefficient for improved convergence
        self.w = 0.7  # Balanced inertia weight
        self.F_base = 0.5  # Increased base mutation factor
        self.CR = 0.9  # Slightly increased crossover rate for better diversity
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.velocities = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))
        # Enhanced Sobol sequence initialization for better coverage of search space
        self.positions = self.lower_bound + (self.upper_bound - self.lower_bound) * np.random.rand(self.population_size, self.dim)
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.adaptive_w_factor = 0.98  # Slightly reduced adaptive factor for gradual change
        self.mutation_adjustment = 0.3  # Balanced parameter for adaptive mutation

    def __call__(self, func):
        function_evaluations = 0
        
        while function_evaluations < self.budget:
            # Evaluate current population
            scores = np.apply_along_axis(func, 1, self.positions)
            function_evaluations += self.population_size
            
            # Update personal bests and global best
            for i in range(self.population_size):
                if scores[i] < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = scores[i]
                    self.personal_best_positions[i] = self.positions[i]
                if scores[i] < self.global_best_score:
                    self.global_best_score = scores[i]
                    self.global_best_position = self.positions[i]
            
            # PSO Update with improved coefficients
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            self.velocities = (self.w * self.velocities +
                               self.c1 * r1 * (self.personal_best_positions - self.positions) +
                               self.c2 * r2 * (self.global_best_position - self.positions))
            self.positions += self.velocities
            self.w *= self.adaptive_w_factor
            
            # Differential Evolution Mutation and Crossover with adaptive mutation scaling
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                F_adaptive = self.F_base + self.mutation_adjustment * (1 - (scores[i] / (self.global_best_score + 1e-8)))
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

            # Ensure positions are within bounds
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

        return self.global_best_position