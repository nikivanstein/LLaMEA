import numpy as np

class HybridDE_APSO_Elitism_AdaptiveMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 40
        self.population = np.random.uniform(-5.0, 5.0, (self.initial_pop_size, dim))
        self.fitness = np.full(self.initial_pop_size, np.inf)
        self.velocity = np.random.uniform(-1.0, 1.0, (self.initial_pop_size, dim))
        self.visited_points = 0
        self.best_particle = None
        self.best_particle_fitness = np.inf

    def __call__(self, func):
        F_min = 0.2
        F_max = 1.0
        CR_min = 0.4
        CR_max = 0.9
        inertia_weight = 0.9

        while self.visited_points < self.budget:
            pop_size = max(5, int(self.initial_pop_size * (1 - (self.visited_points / self.budget))))
            new_population = np.copy(self.population[:pop_size])

            for i in range(pop_size):
                if self.visited_points >= self.budget:
                    break

                CR = CR_min + (CR_max - CR_min) * (1 - (self.visited_points / self.budget))
                F = F_min + np.random.rand() * (F_max - F_min)

                indices = np.random.choice(pop_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(pop_size, 3, replace=False)
                a, b, c = self.population[indices]
                mutant = np.clip(a + F * (b - c), -5.0, 5.0)

                crossover_vector = np.copy(self.population[i])
                for j in range(self.dim):
                    if np.random.rand() < CR:
                        crossover_vector[j] = mutant[j]

                new_fitness = func(crossover_vector)
                self.visited_points += 1

                if new_fitness < self.fitness[i]:
                    new_population[i] = crossover_vector
                    self.fitness[i] = new_fitness

                    if new_fitness < self.best_particle_fitness:
                        self.best_particle_fitness = new_fitness
                        self.best_particle = crossover_vector

            # Update velocities and positions (APSO component)
            cognitive_component = 2.0 * np.random.rand(pop_size, self.dim) * (self.population[:pop_size] - new_population[:pop_size])
            social_component = 2.0 * np.random.rand(pop_size, self.dim) * (self.best_particle - new_population[:pop_size])
            self.velocity[:pop_size] = inertia_weight * self.velocity[:pop_size] + cognitive_component + social_component
            self.population[:pop_size] = np.clip(new_population[:pop_size] + self.velocity[:pop_size], -5.0, 5.0)

            best_idx = np.argmin(self.fitness)
            new_population[0] = self.population[best_idx]

            # Adaptive mutation
            F_max *= 0.95

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]