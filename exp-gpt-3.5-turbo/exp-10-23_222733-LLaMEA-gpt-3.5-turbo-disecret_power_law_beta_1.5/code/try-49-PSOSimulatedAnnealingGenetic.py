import numpy as np

class PSOSimulatedAnnealingGenetic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 20
        self.inertia_weight = 0.5
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.initial_temp = 1.0
        self.final_temp = 0.001
        self.alpha = (self.initial_temp - self.final_temp) / budget
        self.current_temp = self.initial_temp
        self.mutation_rate = 0.1
        self.selection_pressure = 2

    def __call__(self, func):
        def crossover(p1, p2):
            mask = np.random.randint(0, 2, size=self.dim)
            c1 = np.where(mask, p1, p2)
            c2 = np.where(mask, p2, p1)
            return c1, c2

        def mutation(individual):
            mask = np.random.rand(self.dim) < self.mutation_rate
            individual[mask] = np.random.uniform(-5.0, 5.0, np.sum(mask))
            return individual

        def selection(population, fitness):
            fitness_values = np.array([fitness(p) for p in population])
            idx = np.argsort(fitness_values)[:self.swarm_size]
            return [population[i] for i in idx]

        def genetic_algorithm(population, fitness):
            new_population = []
            for _ in range(self.budget):
                selected = selection(population, fitness)
                while len(new_population) < self.swarm_size:
                    p1, p2 = np.random.choice(selected, size=2, replace=False)
                    c1, c2 = crossover(p1, p2)
                    c1 = mutation(c1)
                    c2 = mutation(c2)
                    new_population.extend([c1, c2])
                population = new_population[:self.swarm_size]
                new_population = []
            return population

        def update_position(particle, pbest):
            velocity = particle['velocity']
            position = particle['position']
            new_velocity = self.inertia_weight * velocity + self.cognitive_weight * np.random.rand() * (pbest['position'] - position) + self.social_weight * np.random.rand() * (gbest['position'] - position)
            new_position = position + new_velocity
            return new_position

        def perturb_position(position):
            perturbed_position = np.clip(position + np.random.uniform(-0.5, 0.5, self.dim), -5.0, 5.0)
            return perturbed_position

        def acceptance_probability(energy, new_energy, temperature):
            if new_energy < energy:
                return 1.0
            return np.exp((energy - new_energy) / temperature)

        def simulated_annealing(x, func):
            energy = func(x)
            for _ in range(self.budget):
                new_x = perturb_position(x)
                new_energy = func(new_x)
                if acceptance_probability(energy, new_energy, self.current_temp) > np.random.rand():
                    x = new_x
                    energy = new_energy
            return x
        
        particles = [{'position': np.random.uniform(-5.0, 5.0, self.dim), 'velocity': np.zeros(self.dim)} for _ in range(self.swarm_size)]
        gbest = min(particles, key=lambda p: func(p['position']))

        for _ in range(self.budget):
            for particle in particles:
                particle['position'] = update_position(particle, particle)
                particle['position'] = simulated_annealing(particle['position'], func)
            particles_positions = [particle['position'] for particle in particles]
            particles_positions = genetic_algorithm(particles_positions, func)
            particles = [{'position': pos, 'velocity': np.zeros(self.dim)} for pos in particles_positions]

        gbest = min(particles, key=lambda p: func(p['position']))
        
        return gbest['position']