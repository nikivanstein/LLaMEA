import numpy as np

class EnhancedQuantumMemeticPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.initial_inertia = 0.9  # Adaptive inertia start
        self.final_inertia = 0.4  # Adaptive inertia end
        self.c1 = 2.0  # Enhanced cognitive component
        self.c2 = 2.0  # Enhanced social component
        self.q_influence = 0.1  # Quantum influence factor
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.velocity_clamp = 3.0
        self.eval_count = 0

    def __call__(self, func):
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best = particles.copy()
        personal_best_fitness = np.full(self.pop_size, float('inf'))
        global_best = particles[0].copy()
        global_best_fitness = float('inf')
        
        while self.eval_count < self.budget:
            inertia = self.initial_inertia - (
                (self.initial_inertia - self.final_inertia) * (self.eval_count / self.budget))
            
            for i in range(self.pop_size):
                fitness = func(particles[i])
                self.eval_count += 1

                if fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best[i] = particles[i].copy()
                
                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best = particles[i].copy()

                if self.eval_count >= self.budget:
                    break

            if self.eval_count >= self.budget:
                break

            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            velocities = (inertia * velocities + 
                          self.c1 * r1 * (personal_best - particles) +
                          self.c2 * r2 * (global_best - particles) +
                          self.q_influence * (np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim)) - particles))
            velocities = np.clip(velocities, -self.velocity_clamp, self.velocity_clamp)
            
            particles = particles + velocities
            particles = np.clip(particles, self.lower_bound, self.upper_bound)

            for i in range(self.pop_size):
                if np.random.rand() < self.q_influence:
                    idxs = [idx for idx in range(self.pop_size) if idx != i]
                    a, b, c = np.random.choice(idxs, 3, replace=False)
                    mutant_vector = personal_best[a] + self.q_influence * (personal_best[b] - personal_best[c])
                    mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                    trial_vector = np.where(np.random.rand(self.dim) < self.q_influence,
                                            mutant_vector, particles[i])

                    trial_fitness = func(trial_vector)
                    self.eval_count += 1

                    if trial_fitness < personal_best_fitness[i]:
                        personal_best_fitness[i] = trial_fitness
                        personal_best[i] = trial_vector.copy()

                    if trial_fitness < global_best_fitness:
                        global_best_fitness = trial_fitness
                        global_best = trial_vector.copy()

                    if self.eval_count >= self.budget:
                        break

        return global_best