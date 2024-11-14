import numpy as np

class Improved_PSO_DE_Optimizer:
    def __init__(self, budget, dim, swarm_size=30, p_c=0.8, f=0.5):
        self.budget, self.dim, self.swarm_size, self.p_c, self.f = budget, dim, swarm_size, p_c, f

    def __call__(self, func):
        def fitness(x):
            return func(x)

        particles = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))
        pbest = particles.copy()
        pbest_scores = np.array([fitness(p) for p in pbest])
        gbest_idx = np.argmin(pbest_scores)
        gbest = pbest[gbest_idx].copy()
        gbest_score = pbest_scores[gbest_idx]

        evaluations = self.swarm_size

        while evaluations < self.budget:
            new_global_best_found = False
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = 0.5 * velocities[i] + 2.0 * r1 * (pbest[i] - particles[i]) + 2.0 * r2 * (gbest - particles[i])
                particles[i] += velocities[i]

                if np.random.rand() < self.p_c:
                    idx = np.random.choice(self.swarm_size, 3, replace=False)
                    v = particles[i] + self.f * (particles[idx[0]] - particles[idx[1]] + particles[idx[2]])
                    v_score = fitness(np.clip(v, -5.0, 5.0))

                    if v_score < pbest_scores[i]:
                        pbest[i], pbest_scores[i] = v, v_score

                        if v_score < gbest_score:
                            gbest, gbest_score = v.copy(), v_score
                            new_global_best_found = True

                    evaluations += 1
                    if evaluations >= self.budget:
                        break

            if new_global_best_found:
                particles = np.clip(particles, -5.0, 5.0)

        return gbest