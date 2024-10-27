import numpy as np

class PSOSimulatedAnnealing:
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
    
    def __call__(self, func):
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

        gbest = min(particles, key=lambda p: func(p['position']))
        
        return gbest['position']