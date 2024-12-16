class ImprovedDynamicHybridPSOSA_DE(DynamicHybridPSOSA_DE):
    def __init__(self, budget, dim, num_particles=30, max_iterations=1000, c1=1.5, c2=1.5, initial_temp=100, cooling_rate=0.95, f=0.5, cr=0.9, f_lower=0.1, f_upper=0.9, cr_lower=0.1, cr_upper=0.9, chaos_param=3.9):
        super().__init__(budget, dim, num_particles, max_iterations, c1, c2, initial_temp, cooling_rate, f, cr, f_lower, f_upper, cr_lower, cr_upper)
        self.chaos_param = chaos_param

    def chaotic_map(self, x):
        return logistic.cdf(self.chaos_param * x) * 10.0 - 5.0