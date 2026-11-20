import scipy.integrate as itg
import numpy as np
import matplotlib.pyplot as plt

# Código para definir las ecuaciones de movimiento del sistema

"""
Cálculo de las contribuciones de cada uno de los 
componentes del potencial galáctico en la aceleración.
"""

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

#################################################################################

"""
Definción de las ecuaciones de movimiento en el marco inercial
"""

def f(t, state):
    x, y, z, vx, vy, vz = state
    v = np.array([x,y,z])
    ax, ay, az = acc(v)
    return np.array([vx, vy, vz, ax, ay, az])

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

#################################################################################

"""
Algunas cantidades conservadas:
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
    R_ = np.minimum(R, truncation)
    R_  = max(R, 1e-6)
    
    omega = np.sqrt(x**2 + y**2)

    M_R = m3*(R_/a3)**(2.02)/(1 + (R_/a3)**(1.02))
    term1 = -(1.02)/(1 + (100/a3)**(1.02)) + np.log(1 + (100/a3)**(1.02))
    term2 = -(1.02)/(1 + (R_/a3)**(1.02)) + np.log(1 + (R_/a3)**(1.02))
        
    phi1 = -m1/ np.sqrt(R**2 +b1**2)
    phi2 = -m2 / np.sqrt(omega**2 + (a2 + np.sqrt(z**2 + b2**2))**2)
    phi3 = -M_R/R_ - (m3/(1.02 * a3)) * (term1 - term2)

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

#################################################################################

""" Matrices de rotacion"""

def RV(state, omega, ts):
    """
    Matriz de rotación para una velocidad angular que tiene 
    componente únicamente en el eje z.
    """
    w = omega[2] 
    theta = w * ts

    sin = np.sin(theta)
    cos = np.cos(theta)

    rotated = np.empty_like(state, dtype = float)

    rotated[:, 0] = (np.cos(theta) * state[:, 0] + np.sin(theta) * state[:, 1])
    rotated[:, 1] = (-np.sin(theta) * state[:, 0] + np.cos(theta) * state[:, 1])
    rotated[:, 2] = state[:, 2]

    rotated[:, 3] = sin*(state[:,4] - w*state[:,0]) + cos*(state[:,3] + w*state[:,1])
    rotated[:, 4] = -sin*(state[:,3] + w*state[:,1]) + cos*(state[:,4] - w*state[:,0])
    rotated[:, 5] = state[:, 5]

    return rotated

def RinvV(state, omega, ts):
    """
    Inversa de la matriz de rotación anterior
    """
    w = omega[2] 
    theta = w * ts

    cos = np.cos(theta)
    sin = np.sin(theta)

    static = np.empty_like(state, dtype = float)

    static[:, 0] = (cos * state[:, 0] - sin * state[:, 1])
    static[:, 1] = (sin * state[:, 0] + cos * state[:, 1])
    static[:, 2] = state[:, 2]

    static[:, 3] = -sin*(w*state[:,0] + state[:,4]) + cos*(state[:,3] - w*state[:,1])
    static[:, 4] = -sin*(w*state[:, 1] - state[:,3]) + cos*(w*state[:,0] + state[:, 4])
    static[:, 5] = state[:,5]

    return static