import galpy
from galpy.potential import (plotRotcurve, McMillan17 as mc17)
import matplotlib as plt

plotRotcurve(mc17, label = "Rotation Curve") 
plt.savefig("./RotationCurve.png")
plt.show()
