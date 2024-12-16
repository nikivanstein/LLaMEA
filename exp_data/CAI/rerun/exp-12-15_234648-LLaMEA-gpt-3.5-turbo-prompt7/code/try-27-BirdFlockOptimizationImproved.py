class BirdFlockOptimizationImproved(BirdFlockOptimization):
    def __init__(self, budget, dim, num_birds=20, w=0.5, c1=1.5, c2=1.5, c1_min=0.5, c1_max=2.0, c2_min=0.5, c2_max=2.0):
        super().__init__(budget, dim, num_birds, w, c1, c2)
        self.c1_min = c1_min
        self.c1_max = c1_max
        self.c2_min = c2_min
        self.c2_max = c2_max

    def __call__(self, func):
        def update_velocity(velocity, position, global_best_pos, personal_best_pos, iteration):
            r1, r2 = np.random.rand(), np.random.rand()
            w = self.w * (1.0 - iteration / self.budget)  # Dynamic inertia weight
            c1 = self.c1_min + (self.c1_max - self.c1_min) * (iteration / self.budget)  # Dynamic cognitive coefficient
            c2 = self.c2_min + (self.c2_max - self.c2_min) * (iteration / self.budget)  # Dynamic social coefficient
            return w * velocity + c1 * r1 * (personal_best_pos - position) + c2 * r2 * (global_best_pos - position)
