
import hickle as hkl
import numpy as np

year = 2024
doy = 365

path = f"data\hickle\gim_{year}_hourlyaux.hickle"
data = hkl.load(path)

d = data['data']
print(d[doy-1]["kp_array"])
