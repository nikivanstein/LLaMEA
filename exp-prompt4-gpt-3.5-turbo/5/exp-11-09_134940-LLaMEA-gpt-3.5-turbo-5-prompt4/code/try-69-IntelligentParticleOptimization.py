import numpy as np

class IntelligentParticleOptimization:
    def __init__(self, budget, dim, num_particles=30, inertia_weight=0.5, cognitive_weight=1.5, social_weight=2.0, inertia_decay=0.95):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.inertia_decay = inertia_decay

    def __call__(self, func):
        def initialize_particles():
            return np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))

        def update_particle(particle, personal_best, global_best, epoch):
            inertia_term = self.inertia_weight * particle["velocity"]
            cognitive_term = self.cognitive_weight * np.random.rand() * (personal_best - particle["position"])
            social_term = self.social_weight * np.random.rand() * (global_best - particle["position"])
            new_velocity = inertia_term + cognitive_term + social_term
            new_position = particle["position"] + new_velocity
            return {"position": np.clip(new_position, -5.0, 5.0), "velocity": new_velocity}

        particles = [{"position": pos, "velocity": np.zeros_like(pos)} for pos in initialize_particles()]
        personal_bests = np.copy([particle["position"] for particle in particles])
        global_best = personal_bests[np.argmin([func(p) for p in personal_bests])]

        for epoch in range(self.budget):
            for i, particle in enumerate(particles):
                particles[i] = update_particle(particle, personal_bests[i], global_best, epoch)
                f_val = func(particles[i]["position"])
                if f_val < func(personal_bests[i]):
                    personal_bests[i] = particles[i]["position"]
                if f_val < func(global_best):
                    global_best = particles[i]["position"]
                    
            self.inertia_weight *= self.inertia_decay

        return global_best