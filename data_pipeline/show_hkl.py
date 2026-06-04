
import hickle as hkl

from iono.config import Config

year = 2024
doy = 365

path = Config.data_dir / "hickle" / f"gim_{year}_hourlyaux.hickle"
data = hkl.load(path)

d = data['data']
print(d[doy-1]["kp_array"])
