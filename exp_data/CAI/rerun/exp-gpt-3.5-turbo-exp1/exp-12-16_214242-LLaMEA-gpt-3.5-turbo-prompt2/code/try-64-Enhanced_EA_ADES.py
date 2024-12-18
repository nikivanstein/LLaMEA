class Enhanced_EA_ADES(Improved_EA_ADES):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
    
    def __call__(self, func):
        def crossover(x, trial):
            cr = np.random.normal(self.cr, 0.1)  # Adaptive crossover probability with a small variation
            cr = np.clip(cr, 0.1, 0.9)
            jrand = np.random.randint(self.dim)
            return np.where(np.random.uniform(0, 1, self.dim) < cr, trial, x)
        
        return super().__call__(func)