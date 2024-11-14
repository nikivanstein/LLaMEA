import numpy as np

class CooperativeDualSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 50  # Adjusted swarm size for effectiveness
        self.inertia = 0.5  # Balanced inertia for convergence and exploration
        self.cognitive = 1.5  # Enhanced cognitive component
        self.social = 2.0  # Enhanced social component
        self.global_learning = 0.1  # Add global learning rate
        self.local_learning = 0.3  # Add local learning rate
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initialize swarms
        swarm1 = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        swarm2 = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        velocities1 = np.zeros((self.swarm_size, self.dim))
        velocities2 = np.zeros((self.swarm_size, self.dim))
        personal_best1 = swarm1.copy()
        personal_best2 = swarm2.copy()
        personal_best_values1 = np.array([func(ind) for ind in swarm1])
        personal_best_values2 = np.array([func(ind) for ind in swarm2])
        global_best_idx1 = np.argmin(personal_best_values1)
        global_best_idx2 = np.argmin(personal_best_values2)
        global_best1 = personal_best1[global_best_idx1]
        global_best2 = personal_best2[global_best_idx2]

        evaluations = 2 * self.swarm_size

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break

                # Update velocities with cooperative learning
                r1, r2, r3, r4 = np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand()
                velocities1[i] = (self.inertia * velocities1[i] +
                                  self.cognitive * r1 * (personal_best1[i] - swarm1[i]) +
                                  self.social * r2 * (global_best2 - swarm1[i]) +
                                  self.global_learning * r3 * (global_best1 - swarm1[i]))
                velocities2[i] = (self.inertia * velocities2[i] +
                                  self.cognitive * r1 * (personal_best2[i] - swarm2[i]) +
                                  self.social * r4 * (global_best1 - swarm2[i]) +
                                  self.local_learning * r2 * (global_best2 - swarm2[i]))

                # Update positions
                swarm1[i] += velocities1[i]
                swarm2[i] += velocities2[i]
                swarm1[i] = np.clip(swarm1[i], self.lower_bound, self.upper_bound)
                swarm2[i] = np.clip(swarm2[i], self.lower_bound, self.upper_bound)

                # Evaluate new positions
                f_val1 = func(swarm1[i])
                f_val2 = func(swarm2[i])
                evaluations += 2

                # Update personal bests
                if f_val1 < personal_best_values1[i]:
                    personal_best_values1[i] = f_val1
                    personal_best1[i] = swarm1[i].copy()
                    if f_val1 < personal_best_values1[global_best_idx1]:
                        global_best_idx1 = i
                        global_best1 = personal_best1[global_best_idx1]
                
                if f_val2 < personal_best_values2[i]:
                    personal_best_values2[i] = f_val2
                    personal_best2[i] = swarm2[i].copy()
                    if f_val2 < personal_best_values2[global_best_idx2]:
                        global_best_idx2 = i
                        global_best2 = personal_best2[global_best_idx2]

        combined_best = global_best1 if func(global_best1) < func(global_best2) else global_best2
        return combined_best