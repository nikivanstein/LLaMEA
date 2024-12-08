import numpy as np

class GWOPSOHybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_wolves = 10
        self.num_particles = 20
        self.wolf_alpha = 2.0
        self.wolf_beta = 2.0
        self.particle_c1 = 2.0
        self.particle_c2 = 2.0

    def __call__(self, func):
        def initialize_population():
            wolves = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_wolves, self.dim))
            particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
            return wolves, particles

        def evaluate_population(population):
            return np.array([func(solution) for solution in population])

        def update_wolves(wolves, wolf_fitness, particles, particle_fitness):
            best_idx = np.argmin(wolf_fitness)
            best_solution = wolves[best_idx]

            for i in range(self.num_wolves):
                if i != best_idx:
                    new_solution = wolves[i] + np.random.normal(0, 1, self.dim) * (best_solution - wolves[i])
                    new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
                    if func(new_solution) < wolf_fitness[i]:
                        wolves[i] = new_solution
                        wolf_fitness[i] = func(new_solution)

            return wolves, wolf_fitness

        def update_particles(wolves, wolf_fitness, particles, particle_fitness):
            for i in range(self.num_particles):
                new_velocity = particles[i] + self.particle_c1 * np.random.random() * (particles[i] - particles[i])
                new_position = particles[i] + self.particle_c2 * np.random.random() * (wolves[i % self.num_wolves] - particles[i])
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
                if func(new_position) < particle_fitness[i]:
                    particles[i] = new_position
                    particle_fitness[i] = func(new_position)

            return particles, particle_fitness

        wolves, particles = initialize_population()
        wolf_fitness = evaluate_population(wolves)
        particle_fitness = evaluate_population(particles)

        for _ in range(self.budget - self.budget // 10):
            wolves, wolf_fitness = update_wolves(wolves, wolf_fitness, particles, particle_fitness)
            particles, particle_fitness = update_particles(wolves, wolf_fitness, particles, particle_fitness)

        best_idx = np.argmin(wolf_fitness)
        return wolves[best_idx]