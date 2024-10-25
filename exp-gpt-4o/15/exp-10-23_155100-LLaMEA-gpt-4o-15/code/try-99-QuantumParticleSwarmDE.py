import numpy as np

class QuantumParticleSwarmDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 8 * dim
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.best_personal_positions = np.copy(self.population)
        self.best_personal_fitness = np.full(self.population_size, np.inf)
        self.best_solution = None
        self.best_fitness = float('inf')
        self.f_min = 0.5
        self.f_max = 0.9
        self.cr_min = 0.3
        self.cr_max = 0.9
        self.inertia_weight = 0.5
        self.cognitive_component = 1.5
        self.social_component = 1.5
        self.entropy_threshold = 0.02
        self.chaos_control = 2.1
        self.exploration_probability = 0.15

    def adaptive_parameters(self):
        f = np.random.uniform(self.f_min, self.f_max)
        cr = np.random.uniform(self.cr_min, self.cr_max)
        return f, cr

    def chaotic_map(self, x):
        return (self.chaos_control * x + 0.9) % 1.0

    def mutate(self, target_idx, f):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        chaos_factor = self.chaotic_map(np.random.rand())
        mutant = np.clip(a + f * (b - c) * chaos_factor, self.lower_bound, self.upper_bound)
        return mutant

    def crossover(self, target, mutant, cr):
        crossover = np.random.rand(self.dim) < cr
        if not np.any(crossover):
            crossover[np.random.randint(0, self.dim)] = True
        return np.where(crossover, mutant, target)

    def update_velocity(self, idx):
        inertia = self.inertia_weight * self.velocities[idx]
        cognitive = self.cognitive_component * np.random.rand(self.dim) * (self.best_personal_positions[idx] - self.population[idx])
        social = self.social_component * np.random.rand(self.dim) * (self.best_solution - self.population[idx])
        self.velocities[idx] = inertia + cognitive + social

    def calculate_entropy(self):
        return np.mean(np.std(self.population, axis=0))

    def __call__(self, func):
        eval_count = 0
        fitness = np.full(self.population_size, np.inf)

        while eval_count < self.budget:
            for i in range(self.population_size):
                self.update_velocity(i)
                self.population[i] = np.clip(self.population[i] + self.velocities[i], self.lower_bound, self.upper_bound)

                f, cr = self.adaptive_parameters()
                mutant = self.mutate(i, f)
                trial = self.crossover(self.population[i], mutant, cr)

                if self.calculate_entropy() < self.entropy_threshold:
                    trial = self.population[i] + self.chaotic_map(np.random.randn(self.dim))

                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < self.best_personal_fitness[i]:
                    self.best_personal_positions[i] = trial
                    self.best_personal_fitness[i] = trial_fitness

                if trial_fitness < self.best_fitness:
                    self.best_solution = trial
                    self.best_fitness = trial_fitness

                if eval_count >= self.budget:
                    break

        return self.best_solution, self.best_fitness