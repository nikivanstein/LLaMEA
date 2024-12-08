class DynamicPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 18 if dim <= 10 else 17  # Adjust population size dynamically (35.6% reduction)
        self.inertia_weight = 0.9
        self.cognitive_weight = 2.0
        self.social_weight = 2.0
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.current_evals = 0