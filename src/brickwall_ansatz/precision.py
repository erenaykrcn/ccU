# precision.py
import numpy as np

class PrecisionConfig:
    def __init__(self,
                 real_dtype=np.float64,
                 complex_dtype=np.complex128):
        self.real = real_dtype
        self.complex = complex_dtype

        # Convenience constructors
        self.zero_c = complex_dtype(0)
        self.one_c  = complex_dtype(1)
        self.zero_r = real_dtype(0)
        self.one_r  = real_dtype(1)

# Default: high precision
HP = PrecisionConfig(real_dtype=np.longdouble,
                     complex_dtype=np.clongdouble)

# If you ever want standard precision:
SP = PrecisionConfig()
