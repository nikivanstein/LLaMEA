import numpy as np

class EnhancedHybridPSO_DE_Improved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1_start, self.c1_end = 2.0, 0.5
        self.c2_start, self.c2_end = 0.5, 2.0
        self.w_start, self.w_end = 0.9, 0.4
        self.F_start, self.F_end = 0.5, 0.9
        self.CR_start, self.CR_end = 0.8, 0.95
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_evals = 0

    def __call__(self, func):
        def adapt_parameters():
            progress = self.num_evals / self.budget
            c1 = self.c1_start - progress * (self.c1_start - self.c1_end)
            c2 = self.c2_start + progress * (self.c2_end - self.c2_start)
            w = self.w_start - progress * (self.w_start - self.w_end)
            F = self.F_start + progress * (self.F_end - self.F_start)
            CR = self.CR_start + progress * (self.CR_end - self.CR_start)
            return c1, c2, w, F, CR

        def opposition_based_learning(swarm):
            return self.lower_bound + self.upper_bound - swarm

        # Initialize swarm
        swarm = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(swarm)
        personal_best_scores = np.full(self.population_size, np.inf)
        global_best_position = None
        global_best_score = np.inf

        while self.num_evals < self.budget:
            c1, c2, w, F, CR = adapt_parameters()

            # Evaluate current swarm
            for i in range(self.population_size):
                if self.num_evals >= self.budget:
                    break
                score = func(swarm[i])
                self.num_evals += 1

                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = swarm[i]

                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = swarm[i]

            # PSO update
            for i in range(self.population_size):
                inertia = w * velocities[i]
                cognitive = c1 * np.random.random(self.dim) * (personal_best_positions[i] - swarm[i])
                social = c2 * np.random.random(self.dim) * (global_best_position - swarm[i])
                velocities[i] = inertia + cognitive + social

                # Update position
                swarm[i] = swarm[i] + velocities[i]
                # Ensure bounds
                swarm[i] = np.clip(swarm[i], self.lower_bound, self.upper_bound)

            # DE-inspired mutation and crossover with opposition-based learning
            opposite_swarm = opposition_based_learning(swarm)
            for i in range(self.population_size):
                if self.num_evals >= self.budget:
                    break
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = personal_best_positions[indices]
                mutant = a + F * (b - c)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                trial = np.copy(swarm[i])
                for j in range(self.dim):
                    if np.random.rand() < CR:
                        trial[j] = mutant[j]

                # Evaluate both the trial and the opposite
                trial_score = func(trial)
                self.num_evals += 1

                opposite_trial_score = func(opposite_swarm[i])
                self.num_evals += 1

                # Choose better between trial and opposite
                if opposite_trial_score < trial_score:
                    trial_score = opposite_trial_score
                    trial = opposite_swarm[i]

                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial
                    if trial_score < global_best_score:
                        global_best_score = trial_score
                        global_best_position = trial

        return global_best_position