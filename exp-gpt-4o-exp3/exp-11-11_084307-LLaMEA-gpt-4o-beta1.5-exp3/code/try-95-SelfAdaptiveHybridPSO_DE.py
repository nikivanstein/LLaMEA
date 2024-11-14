import numpy as np

class SelfAdaptiveHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_evals = 0

    def __call__(self, func):
        def random_init_population():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

        def adapt_parameters(iteration):
            progress = iteration / self.budget
            w = 0.9 - progress * (0.9 - 0.4)
            c1 = 2.5 - progress * (2.5 - 0.5)
            c2 = 0.5 + progress * (2.0 - 0.5)
            F = 0.5 + progress * (0.9 - 0.5)
            CR = 0.8 + progress * (0.95 - 0.8)
            local_search_prob = 0.1 + progress * (0.5 - 0.1)
            return w, c1, c2, F, CR, local_search_prob

        swarm = random_init_population()
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(swarm)
        personal_best_scores = np.full(self.population_size, np.inf)
        global_best_position = None
        global_best_score = np.inf

        initial_velocity_clamp = (self.upper_bound - self.lower_bound) / 2.0

        while self.num_evals < self.budget:
            w, c1, c2, F, CR, local_search_prob = adapt_parameters(self.num_evals)

            for i in range(self.population_size):
                if self.num_evals >= self.budget:
                    break
                score = func(swarm[i])
                self.num_evals += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = swarm[i]

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = swarm[i]

            for i in range(self.population_size):
                inertia = w * velocities[i]
                cognitive = c1 * np.random.random(self.dim) * (personal_best_positions[i] - swarm[i])
                social = c2 * np.random.random(self.dim) * (global_best_position - swarm[i])
                velocities[i] = inertia + cognitive + social
                clamp = initial_velocity_clamp * (1 - self.num_evals / self.budget)
                velocities[i] = np.clip(velocities[i], -clamp, clamp)

                swarm[i] += velocities[i]
                swarm[i] = np.clip(swarm[i], self.lower_bound, self.upper_bound)

            for i in range(self.population_size):
                if self.num_evals >= self.budget:
                    break
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = personal_best_positions[indices]
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

                if np.random.rand() < local_search_prob:
                    local_trial = trial + np.random.normal(0, 0.1, self.dim)
                    local_trial = np.clip(local_trial, self.lower_bound, self.upper_bound)
                    local_trial_score = func(local_trial)
                    self.num_evals += 1

                    if local_trial_score < personal_best_scores[i]:
                        personal_best_scores[i] = local_trial_score
                        personal_best_positions[i] = local_trial
                        if local_trial_score < global_best_score:
                            global_best_score = local_trial_score
                            global_best_position = local_trial

        return global_best_position