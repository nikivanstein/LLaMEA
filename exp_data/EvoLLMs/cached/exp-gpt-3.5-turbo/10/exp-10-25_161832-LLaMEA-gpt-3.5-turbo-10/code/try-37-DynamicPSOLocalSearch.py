import numpy as np

class DynamicPSOLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = 30
        self.c1 = 1.5
        self.c2 = 1.5
        self.max_velocity = 0.2
        self.local_search_prob = 0.1

    def __call__(self, func):
        def initialize_particles():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))

        def update_velocity(particle, global_best):
            return np.clip(particle["velocity"] + self.c1 * np.random.rand() * (particle["best"] - particle["position"]) + self.c2 * np.random.rand() * (global_best - particle["position"]), -self.max_velocity, self.max_velocity)

        def update_position(particle):
            new_position = particle["position"] + particle["velocity"]
            return np.clip(new_position, self.lower_bound, self.upper_bound)

        particles = [{"position": p, "velocity": np.zeros(self.dim), "best": p, "fitness": func(p)} for p in initialize_particles()]
        global_best = min(particles, key=lambda x: x["fitness"])["position"]

        for _ in range(self.budget):
            for particle in particles:
                new_velocity = update_velocity(particle, global_best)
                new_position = update_position({"position": particle["position"], "velocity": new_velocity, "best": particle["best"], "fitness": particle["fitness"]})
                new_fitness = func(new_position)
                if new_fitness < particle["fitness"]:
                    particle["position"] = new_position
                    particle["fitness"] = new_fitness
                    if new_fitness < func(particle["best"]):
                        particle["best"] = new_position
                if np.random.rand() < self.local_search_prob:  # Adjusted line: Changed the probability for local search
                    local_search_position = np.clip(particle["position"] + np.random.normal(0, 0.1, self.dim), self.lower_bound, self.upper_bound)
                    local_search_fitness = func(local_search_position)
                    if local_search_fitness < particle["fitness"]:
                        particle["position"] = local_search_position
                        particle["fitness"] = local_search_fitness
                        if local_search_fitness < func(particle["best"]):
                            particle["best"] = local_search_position

            global_best = min(particles, key=lambda x: x["fitness"])["position"]

        return min(particles, key=lambda x: x["fitness"])["position"]