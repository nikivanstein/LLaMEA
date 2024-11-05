import numpy as np

class AdaptiveMultiSwarmHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 20
        self.w = 0.7
        self.c1 = 1.5
        self.c2 = 1.5
        self.F = 0.8
        self.CR = 0.9
        self.local_search_prob = 0.1
        self.adaptive_pop_size = self.pop_size
        self.num_swarms = 3

    def __call__(self, func):
        np.random.seed(42)
        swarms = [np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim)) for _ in range(self.num_swarms)]
        velocities = [np.random.uniform(-1, 1, (self.pop_size, self.dim)) for _ in range(self.num_swarms)]
        personal_bests = [swarm.copy() for swarm in swarms]
        personal_best_values = [np.array([func(ind) for ind in swarm]) for swarm in swarms]
        global_best_indices = [np.argmin(p_values) for p_values in personal_best_values]
        global_bests = [personal_bests[i][global_best_indices[i]] for i in range(self.num_swarms)]
        global_best_values = [personal_best_values[i][global_best_indices[i]] for i in range(self.num_swarms)]
        
        evaluations = self.pop_size * self.num_swarms

        while evaluations < self.budget:
            for swarm_idx in range(self.num_swarms):
                self.w = 0.9 - 0.5 * (evaluations / self.budget)
                self.c1 = 2.0 * (1 - evaluations / self.budget)
                self.c2 = 2.0 * (evaluations / self.budget)

                r1 = np.random.rand(self.pop_size, self.dim)
                r2 = np.random.rand(self.pop_size, self.dim)
                velocities[swarm_idx] = self.w * velocities[swarm_idx] + self.c1 * r1 * (personal_bests[swarm_idx] - swarms[swarm_idx]) + self.c2 * r2 * (global_bests[swarm_idx] - swarms[swarm_idx])
                swarms[swarm_idx] = np.clip(swarms[swarm_idx] + velocities[swarm_idx], self.lower_bound, self.upper_bound)

                fitness = np.array([func(ind) for ind in swarms[swarm_idx]])
                evaluations += self.pop_size

                better_mask = fitness < personal_best_values[swarm_idx]
                personal_bests[swarm_idx][better_mask] = swarms[swarm_idx][better_mask]
                personal_best_values[swarm_idx][better_mask] = fitness[better_mask]

                current_global_best_index = np.argmin(personal_best_values[swarm_idx])
                current_global_best_value = personal_best_values[swarm_idx][current_global_best_index]
                if current_global_best_value < global_best_values[swarm_idx]:
                    global_bests[swarm_idx] = personal_bests[swarm_idx][current_global_best_index]
                    global_best_values[swarm_idx] = current_global_best_value
                
                # Fitness sharing and DE operations
                for i in range(self.pop_size):
                    if evaluations >= self.budget:
                        break
                    idxs = np.random.choice([idx for idx in range(self.pop_size) if idx != i], 3, replace=False)
                    a, b, c = swarms[swarm_idx][idxs]
                    mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)

                    cross_points = np.random.rand(self.dim) < self.CR
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    trial = np.where(cross_points, mutant, swarms[swarm_idx][i])

                    trial_fitness = func(trial)
                    evaluations += 1
                    if trial_fitness < fitness[i]:
                        swarms[swarm_idx][i] = trial
                        fitness[i] = trial_fitness
                        if trial_fitness < global_best_values[swarm_idx]:
                            global_bests[swarm_idx] = trial
                            global_best_values[swarm_idx] = trial_fitness

                # Diversity preservation through random restart
                if evaluations % (self.pop_size * 10) == 0:
                    diversity = np.std(swarms[swarm_idx])
                    if diversity < 0.1:
                        swarms[swarm_idx] = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
                        velocities[swarm_idx] = np.random.uniform(-1, 1, (self.pop_size, self.dim))

        overall_best_value = min(global_best_values)
        best_index = global_best_values.index(overall_best_value)
        return global_bests[best_index], overall_best_value