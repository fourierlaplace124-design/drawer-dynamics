import numpy as np
from scipy.integrate import quad

class Mechanics: 
    def Sliding_dist(L1,L2,theta_1,theta_2):
        if theta_1 and theta_2 >= 0 and theta_1 and theta_2 <= 90:
            tan_1 = np.tan(np.radians(theta_1))
            tan_2 = np.tan(np.radians(theta_2))
            cot_1 = 1/tan_1
            d = L2 - L1*cot_1
            if d >= 0:
                real_d = d
            else:
                print("No sliding occurs")
                real_d = 0
        else:
            raise ValueError("Angles must be between 0 and 90 degrees")
        return real_d
    
    
    
