import numpy as np

class MultiSwarmAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = int(10 + 2 * np.sqrt(dim))
        self.num_swarms = 3
        self.position = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_swarms, self.swarm_size, dim))
        self.velocity = np.zeros((self.num_swarms, self.swarm_size, dim))
        self.personal_best_position = np.copy(self.position)
        self.personal_best_score = np.full((self.num_swarms, self.swarm_size), float('inf'))
        self.global_best_position = np.copy(self.position[:, 0, :])
        self.global_best_score = np.full(self.num_swarms, float('inf'))
        self.func_evaluations = 0
        self.global_best_overall_position = None
        self.global_best_overall_score = float('inf')
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.7
        self.adaptive_freq = 50

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            for swarm in range(self.num_swarms):
                for i in range(self.swarm_size):
                    score = func(self.position[swarm, i])
                    self.func_evaluations += 1

                    if score < self.personal_best_score[swarm, i]:
                        self.personal_best_score[swarm, i] = score
                        self.personal_best_position[swarm, i] = self.position[swarm, i]

                    if score < self.global_best_score[swarm]:
                        self.global_best_score[swarm] = score
                        self.global_best_position[swarm] = self.position[swarm, i]

                    if score < self.global_best_overall_score:
                        self.global_best_overall_score = score
                        self.global_best_overall_position = self.position[swarm, i]

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                for i in range(self.swarm_size):
                    self.velocity[swarm, i] = (self.w * self.velocity[swarm, i] +
                                               self.c1 * r1 * (self.personal_best_position[swarm, i] - self.position[swarm, i]) +
                                               self.c2 * r2 * (self.global_best_position[swarm] - self.position[swarm, i]))
                    self.position[swarm, i] = np.clip(self.position[swarm, i] + self.velocity[swarm, i], self.lower_bound, self.upper_bound)

            if self.func_evaluations % self.adaptive_freq == 0:
                self.c1, self.c2 = (2.5 - 1.5 * self.func_evaluations / self.budget), (1.5 + 1.5 * self.func_evaluations / self.budget)
                self.w = 0.9 - 0.4 * self.func_evaluations / self.budget

        return self.global_best_overall_position