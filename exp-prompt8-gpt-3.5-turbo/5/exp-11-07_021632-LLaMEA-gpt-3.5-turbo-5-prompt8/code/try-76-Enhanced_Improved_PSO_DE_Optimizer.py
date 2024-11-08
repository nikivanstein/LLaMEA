import numpy as np

class Enhanced_Improved_PSO_DE_Optimizer:
    def __init__(self, budget, dim, swarm_size=30, p_c=0.8, f=0.5):
        self.budget, self.dim, self.swarm_size, self.p_c, self.f = budget, dim, swarm_size, p_c, f

    def __call__(self, func):
        particles = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
        pbest = particles.copy()
        pbest_scores = np.apply_along_axis(func, 1, pbest)
        gbest_idx = np.argmin(pbest_scores)
        gbest = pbest[gbest_idx].copy()
        gbest_score = pbest_scores[gbest_idx]

        evaluations = self.swarm_size

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(), np.random.rand()
                pbest_i, gbest_i = pbest[i], gbest

                particles[i] += 0.5 * (particles[i] - pbest_i) + 2.0 * r1 * (pbest_i - particles[i]) + 2.0 * r2 * (gbest_i - particles[i])

                if np.random.rand() < self.p_c:
                    idx = np.random.choice(self.swarm_size, 3, replace=False)
                    mutant = particles[idx]
                    v = particles[i] + self.f * (mutant[0] - mutant[1] + mutant[2])
                    v_score = func(np.clip(v, -5.0, 5.0))

                    if v_score < pbest_scores[i]:
                        pbest[i], pbest_scores[i] = v, v_score

                        if v_score < gbest_score:
                            gbest, gbest_score = v.copy(), v_score

                    evaluations += 1
                    if evaluations >= self.budget:
                        break

        return gbest