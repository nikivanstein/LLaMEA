class BirdFlockOptimizationImproved:
    def __init__(self, budget, dim, num_birds=20, w=0.5, c1=1.5, c2=1.5, adaptive_c=False, c_min=0.5, c_max=2.0):
        self.budget = budget
        self.dim = dim
        self.num_birds = num_birds
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.adaptive_c = adaptive_c
        self.c_min = c_min
        self.c_max = c_max

    def __call__(self, func):
        def update_velocity(velocity, position, global_best_pos, personal_best_pos, iteration):
            r1, r2 = np.random.rand(), np.random.rand()
            w = self.w * (1.0 - iteration / self.budget)  # Dynamic inertia weight
            if self.adaptive_c:
                c1 = self.c_min + (self.c_max - self.c_min) * (iteration / self.budget)  # Adaptive acceleration coefficient
                c2 = self.c_min + (self.c_max - self.c_min) * (iteration / self.budget)  # Adaptive acceleration coefficient
            else:
                c1, c2 = self.c1, self.c2
            return w * velocity + c1 * r1 * (personal_best_pos - position) + c2 * r2 * (global_best_pos - position)

        # Rest of the code remains the same