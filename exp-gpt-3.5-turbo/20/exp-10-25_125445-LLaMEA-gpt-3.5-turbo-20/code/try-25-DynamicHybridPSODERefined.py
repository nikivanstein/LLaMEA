import numpy as np

class DynamicHybridPSODERefined(DynamicHybridPSODE):
    def adapt_parameters(self, func, gbest_val):
        improvement_threshold = 0.1
        if np.random.rand() < 0.2:
            if gbest_val < improvement_threshold:
                self.w = np.clip(self.w * 1.1, self.min_w, self.max_w)
                self.c1 = np.clip(self.c1 * 1.1, self.min_c, self.max_c)
                self.c2 = np.clip(self.c2 * 1.1, self.min_c, self.max_c)
                self.f = np.clip(self.f * 1.1, self.min_f, self.max_f)
                self.cr = np.clip(self.cr * 1.1, self.min_cr, self.max_cr)
            else:
                self.w = np.clip(self.w * 0.9, self.min_w, self.max_w)
                self.c1 = np.clip(self.c1 * 0.9, self.min_c, self.max_c)
                self.c2 = np.clip(self.c2 * 0.9, self.min_c, self.max_c)
                self.f = np.clip(self.f * 0.9, self.min_f, self.max_f)
                self.cr = np.clip(self.cr * 0.9, self.min_cr, self.max_cr)