import scipy.integrate as itg
import numpy as np
import matplotlib.pyplot as plt

"""
Condiciones iniciales del sistema
"""

kpc = 1.0                 # already in code length units
kms_to_code = 1.0/10.0    # 1 code velocity unit = 10 km/s

x0, y0_, z0 = 8.0, 0.0, 0.0 #0.021         # kpc: R ~ 8 kpc, slightly above the plane
vx0, vy0, vz0 = 0.0*kms_to_code, 120.0*kms_to_code, 0.0*kms_to_code  # km/s -> code units
state0 = np.array([x0, y0_, z0, vx0, vy0, vz0])

# Integration will be till 5 Gyrs
time_unit_Gyr = 0.0978
t_end_Gyr = 5
t_end = t_end_Gyr / time_unit_Gyr  # Code units

h = 0.00005
number_steps = int(t_end/h)

""" Algunas cantidades conservadas:
    - E
    - Lz

Algunas otras cantidades de interés: 
    - El potencial
    - La energía cinética
    - El momento angular (vector)
    - Velocidad circular (radial)
"""

def pot(v):
    m1, b1 = 606., 0.3873
    m2, a2, b2 = 3690., 5.3178, .25
    m3, a3 = 4615., 12.0

    x, y, z = v[0], v[1], v[2]
    R = np.sqrt(x**2 + y**2 + z**2)
    truncation = 100
    R = np.minimum(R, truncation)
    R  = max(R, 1e-6)
    
    omega = np.sqrt(x**2 + y**2)

    M_R = m3*(R/a3)**(2.02)/(1 + (R/a3)**(1.02))
    term1 = -(1.02)/(1 + (100/a3)**(1.02)) + np.log(1 + (100/a3)**(1.02))
    term2 = -(1.02)/(1 + (R/a3)**(1.02)) + np.log(1 + (R/a3)**(1.02))
        
    phi1 = -m1/ np.sqrt(R**2 +b1**2)
    phi2 = -m2 / np.sqrt(omega**2 + (a2 + np.sqrt(z**2 + b2**2))**2)
    phi3 = -M_R/R - (m3/(1.02 * a3)) * (term1 - term2)

    phi = phi1 + phi2 + phi3
    return phi

def T(v):
    vx, vy, vz = v[3], v[4], v[5]
    t = (1/2)*(vx**2 + vy**2 + vz**2)
    return t
    
def l(v):
    r = np.array([v[0], v[1], v[2]])
    v = np.array([v[3], v[4], v[5]])
    l = np.cross(r, v)
    return l

def vel_c(v, code_units = True):
    ac = acc(v)
    x, y, z = v[0], v[1], v[2]
    if code_units == False:
        vel = np.sqrt(-ac[0]*x - ac[1]*y - ac[2]*z)*10
    else:
        vel = np.sqrt(-ac[0]*x - ac[1]*y - ac[2]*z)
    return vel

"""
Función para el cálculo de los errores relativos entre
valores inciales y finales de cantidades conservadas. 
"""
def err(v1, v2): 
    err = np.abs(v1 - v2)/np.abs(v2)
    return err

""" Cálculo de las contribuciones de cada uno de las componentes del potencial galáctico en la aceleración. """

M3 = 4615.
a3 = 12.

def halo(x, R):
    a = -(1.02*M3*(x/R))*((R/a3)**2.02)
    b = R*(1+(R/a3)**1.02)
        
    c = M3 * x * ((R/a3)**(1.02)) / (a3 * R**2)
    d = 1 + (R/a3)**(1.02)
        
    halo = c / d
    return halo

def acc(v): 
    x, y, z = v[0], v[1], v[2]
    R = np.sqrt(x**2 + y**2 + z**2) # kpc units
    truncation = 100
    R = np.minimum(R, truncation)
    R  = max(R, 1e-6)
    
    # Central mass
    M1 = 606.
    b1 = 0.3873
    
    dphi_x1 =  M1*x/(x**2 + y**2 + z**2 + b1**2)**(3/2)
    dphi_y1 =  M1*y/(x**2 + y**2 + z**2 + b1**2)**(3/2)
    dphi_z1 =  M1*z/(x**2 + y**2 + z**2 + b1**2)**(3/2) 
    
    # Disk component
    M2 = 3690.
    a2 = 5.3178
    b2 = 0.25
    
    dphi_x2 = M2*x / (x**2 + y**2 + (a2 + np.sqrt(z**2 + b2**2))**2)**(3/2)
    dphi_y2 = M2*y / (x**2 + y**2 + (a2 + np.sqrt(z**2 + b2**2))**2)**(3/2)
    dphi_z2 = M2*z*(a2 + np.sqrt(z**2 + b2**2))/(np.sqrt(z**2 + b2**2)*(np.sqrt(x**2 + y**2 + (a2 + np.sqrt(z**2 + b2**2))**2))**3)
    
    # Halo
    
    dphi_x3 = halo(x, R)
    dphi_y3 = halo(y, R)
    dphi_z3 = halo(z, R)
    
    dphi_x = dphi_x1 + dphi_x2 + dphi_x3
    dphi_y = dphi_y1 + dphi_y2 + dphi_y3
    dphi_z = dphi_z1 + dphi_z2 + dphi_z3

    acx, acy, acz = -dphi_x, -dphi_y, -dphi_z
    return np.array([acx, acy, acz])

"""
Definición de la ecuación de movimiento a integrar en el marco no inercial
"""

omega = np.array([0., 0., -2.3])

def f_nonin(t, state):

    omega_z = omega[2]

    """
    Cantidades en el marco de referencia no inercial
    """
    x, y, z, vx, vy, vz = state # Estado en el marco de referencia NO inercial
    r = np.array([x,y,z]) # Vector de posición
    ax, ay, az = acc(r) # Aceleraciones, por el potencial, en el marco de referencia inercial que coinciden con el no inercial

    """
    Aceleraciones por las fuerzas ficticias
    """
    cor_x, cor_y = 2*omega_z*vy, -2*omega_z*vx
    cent_x, cent_y = (omega_z**2)*x, (omega_z**2)*y
    ficx, ficy = cor_x + cent_x, cor_y + cent_y

    """
    Cálculo de las aceleraciones en el marco no inercial
    """
    n_inx, n_iny, n_inz = ax + ficx, ay + ficy, az
    
    return np.array([vx, vy, vz, n_inx, n_iny, n_inz])

vx0_, vy0_, vz0_ = vx0 + omega[2]*y0_, vy0 - omega[2]*x0, vz0
state0_ = np.array([x0, y0_, z0, vx0_, vy0_, vz0_])

solver = itg.RK45(f_nonin, 0, state0_, t_end, max_step = 0.0005, rtol = 1e-10)

ts, soly = [solver.t], [solver.y.copy()]

while solver.status == 'running':
    solver.step()                # Advances one adaptive RK45 step
    ts.append(solver.t)          # These are the time steps
    soly.append(solver.y.copy()) # These is the solution for each step

soly = np.array(soly)

# Relative error in initial and final energy

e0 = T(state0_) + pot(state0_)
ef = T(soly[-1]) + pot(soly[-1])

error = err(e0, ef)
print(error)

l0 = l(state0_); lz_0 = l0[2]
lf = l(soly[-1]); lz_f = lf[2]

error_lz = err(lz_0, lz_f)
print(error_lz)

def JacInt(E, omega, l):
    Ej = E - np.dot(omega, l)
    return Ej

ej_0 = JacInt(e0, omega, l0)
ej_f = JacInt(ef, omega, lf)

error_ej = err(ej_0, ej_f)
print(error_ej)