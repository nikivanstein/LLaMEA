import numpy as np

class Refined_PSO_SA_Optimizer:
    def __init__(self, budget, dim, num_particles=30, max_iterations=1000, inertia_weight=0.7, cognitive_weight=1.4, social_weight=1.4, initial_temperature=100, cooling_rate=0.95):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate

    def __call__(self, func):
        def refined_pso_sa_helper():
            # PSO initialization with refined strategy
            # PSO optimization loop with refined strategy
            # SA initialization with refined strategy
            # SA optimization loop with refined strategy

        return refined_pso_sa_helper()