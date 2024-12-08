import numpy as np

class EnhancedHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1 = 1.49445  # Cognitive component
        self.c2 = 1.49445  # Social component
        self.w_start = 0.9  # Start inertia weight
        self.w_end = 0.4    # End inertia weight
        self.F_start = 0.5
        self.F_end = 0.9
        self.CR_start = 0.8
        self.CR_end = 0.95
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_evals = 0

    def __call__(self, func):
        def adapt_parameters():
            progress = self.num_evals / self.budget
            w = self.w_start - progress * (self.w_start - self.w_end)
            F = self.F_start + progress * (self.F_end - self.F_start)
            CR = self.CR_start + progress * (self.CR_end - self.CR_start)
            return w, F, CR

        def opposition_based_learning(swarm):
            opposite_swarm = self.lower_bound + self.upper_bound - swarm
            opposite_swarm = np.clip(opposite_swarm, self.lower_bound, self.upper_bound)
            return opposite_swarm

        # Initialize swarm
        swarm = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(swarm)
        personal_best_scores = np.full(self.population_size, np.inf)
        global_best_position = None
        global_best_score = np.inf

        initial_velocity_clamp = (self.upper_bound - self.lower_bound) / 2.0

        # Introduce a phase-based approach
        def switch_phase(iteration):
            return (iteration < 0.5 * self.budget)

        while self.num_evals < self.budget:
            w, F, CR = adapt_parameters()
            velocity_clamp = initial_velocity_clamp * (1 - self.num_evals / self.budget)
            exploration_phase = switch_phase(self.num_evals)

            # Evaluate current swarm and opposition-based learning
            if self.num_evals < self.budget // 2:
                opposite_swarm = opposition_based_learning(swarm)
                for i in range(self.population_size):
                    if self.num_evals >= self.budget:
                        break
                    score = func(swarm[i])
                    opposite_score = func(opposite_swarm[i])
                    self.num_evals += 2

                    if score < personal_best_scores[i]:
                        personal_best_scores[i] = score
                        personal_best_positions[i] = swarm[i]

                    if opposite_score < personal_best_scores[i]:
                        personal_best_scores[i] = opposite_score
                        personal_best_positions[i] = opposite_swarm[i]

                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = swarm[i]

                    if opposite_score < global_best_score:
                        global_best_score = opposite_score
                        global_best_position = opposite_swarm[i]

            # PSO update with adaptive inertia and dynamic velocity clamping
            for i in range(self.population_size):
                inertia = w * velocities[i]
                cognitive = self.c1 * np.random.random(self.dim) * (personal_best_positions[i] - swarm[i])
                social = self.c2 * np.random.random(self.dim) * (global_best_position - swarm[i])
                velocities[i] = inertia + cognitive + social
                velocities[i] = np.clip(velocities[i], -velocity_clamp, velocity_clamp)

                swarm[i] += velocities[i]
                swarm[i] = np.clip(swarm[i], self.lower_bound, self.upper_bound)

            # DE-inspired mutation and crossover with a focus on the exploration phase
            for i in range(self.population_size):
                if self.num_evals >= self.budget:
                    break
                if exploration_phase:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    a, b, c = personal_best_positions[indices]
                else:
                    candidates = np.random.choice(self.population_size, 5, replace=False)
                    best_candidate = min(candidates, key=lambda idx: personal_best_scores[idx])
                    a, b, c = personal_best_positions[[best_candidate] + list(np.random.choice(candidates, 2, replace=False))]

                mutant = a + F * (b - c)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                trial = np.copy(swarm[i])
                for j in range(self.dim):
                    if np.random.rand() < CR or j == np.random.randint(0, self.dim):
                        trial[j] = mutant[j]

                trial_score = func(trial)
                self.num_evals += 1

                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial
                    if trial_score < global_best_score:
                        global_best_score = trial_score
                        global_best_position = trial

        return global_best_position