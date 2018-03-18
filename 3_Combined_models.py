import fenics as fn
fn.parameters["form_compiler"]["representation"] = "uflacs"
fn.parameters["form_compiler"]["cpp_optimize"] = True

# create mesh
mesh_size = mesh_width, mesh_height = 6.0, 6.0
mesh = fn.RectangleMesh(fn.Point(-mesh_width/2, 0.0),
                        fn.Point( mesh_width/2, mesh_height),
                        25, 25)  # square elements

# time-related variables
t = 0.0
dt = 0.3
TFinal = 600.0
frequency = 10

# normals
nn = fn.FacetNormal(mesh)

# element spaces
Vh = fn.VectorElement("CG", mesh.ufl_cell(), 2)
Zh = fn.FiniteElement("CG", mesh.ufl_cell(), 1)
Qh = fn.FiniteElement("CG", mesh.ufl_cell(), 2)
Mh = fn.FiniteElement("CG", mesh.ufl_cell(), 1)
Wh = fn.TensorElement("CG", mesh.ufl_cell(), 1)

# function spaces
Vhf = fn.FunctionSpace(mesh, Vh)
Zhf = fn.FunctionSpace(mesh, Zh)
Qhf = fn.FunctionSpace(mesh, Qh)
Mhf = fn.FunctionSpace(mesh, Mh)
Whf = fn.FunctionSpace(mesh, Wh)

Hh = fn.FunctionSpace(mesh, fn.MixedElement([Vh,Zh,Qh]))
Nh = fn.FunctionSpace(mesh, fn.MixedElement([Mh, Mh]))

# functions and test functions
u, phi, p = fn.TrialFunctions(Hh)
v, psi, q = fn.TestFunctions(Hh)

E, n = fn.TrialFunctions(Nh)
F, m = fn.TestFunctions(Nh)

# pvd files
pvdU   = fn.File(mesh.mpi_comm(), "Output/3_Combined_models/u.pvd")
pvdPHI = fn.File(mesh.mpi_comm(), "Output/3_Combined_models/phi.pvd")
pvdP   = fn.File(mesh.mpi_comm(), "Output/3_Combined_models/p.pvd")
pvdE   = fn.File(mesh.mpi_comm(), "Output/3_Combined_models/E.pvd")
pvdN   = fn.File(mesh.mpi_comm(), "Output/3_Combined_models/n.pvd")

# constants for poroelasticity model (taken from footing-scaled-BCs.py)
EE     = fn.Constant(3.0e4)
nu     = fn.Constant(0.4995)
lmbda  = EE*nu/((1.+nu)*(1.-2.*nu)) 
mu     = EE/(2.*(1.+nu))

c0     = fn.Constant(1.0e-3)
kappa  = fn.Constant(1.0e-4)
alpha  = fn.Constant(0.1)
sigma0 = fn.Constant(1.5e4)
eta    = fn.Constant(1.0)

# constants for reaction-diffusion model
# (taken from ReactionDiffusion_Karma-edited.py)
diffScale = fn.Constant(1e-3)
D0    = 1.1*diffScale
tauE  = fn.Constant(2.5)
taun  = fn.Constant(250.0)
Re    = fn.Constant(1.0)
M     = fn.Constant(5.0)
beta  = fn.Constant(0.008)
Estar = fn.Constant(1.5415)
Eh    = fn.Constant(3.0)
En    = fn.Constant(1.0)

f_0 = fn.Constant((0.0, 1.0))
c_1 = fn.Constant(1e5)

# functions
# body force
f = fn.Constant((0.0, 0.0))  # no body force (gravity)

# dirichlet boundary conditions
u_g = fn.Constant((0.0, 0.0))  # 0 displacement on walls
p_g = fn.Constant(0.0)  # 0 pressure on free surface
h_g = fn.Constant((0.0, -sigma0))  # force applied at foot
#h_g = fn.Expression(("0","-sigma0*t"), sigma0=sigma0, t=0.0, degree=3)

# symmetric strain tensor
def strain(u):
    return fn.sym(fn.grad(u))

# functions - reaction-diffusion
def R(n):
    return (1.0 - (1.0 - fn.exp(-Re)) * n ) / (1.0 - fn.exp(-Re))

def heaviside(r):
    return 0.5 * (fn.sign(r) + 1.0)

def RD_g(E, n):
    return 1.0/taun * (R(n) * heaviside(E - En) \
                       - (1.0 - heaviside(E - En)) * n)

def RD_f(E, n):
    return 1.0/tauE * (-E + (Estar - pow(n, M)) \
                       * (1.0 - fn.tanh(E - Eh)) * pow(E, 2) * 0.5)

def D(E, sigma):  # this is important: it's where the coupling occurs
    return fn.Identity(2) * D0 * (1 + D0*E) + pow(D0, 2) * sigma

# boundaries
mesh_boundary = fn.MeshFunctionSizet(mesh, 1)
mesh_boundary.set_all(0)

# section of boundary that is free
free_boundary = fn.CompiledSubDomain(
        "(near(x[1], mesh_height) || near(x[0], mesh_width/2)) && on_boundary",
        mesh_height=mesh_height, mesh_width=mesh_width)

# section of boundary that is a rigid wall
wall_boundary = fn.CompiledSubDomain(
        "(near(x[1], 0.0) || near(x[0], -mesh_width/2)) && on_boundary",
        mesh_width=mesh_width)

# mark boundaries
free_boundary.mark(mesh_boundary, 31)
wall_boundary.mark(mesh_boundary, 32)

# ds, for integrating
ds = fn.Measure("ds", subdomain_data=mesh_boundary)

# boundary conditions
bcU  = fn.DirichletBC(Hh.sub(0), u_g, mesh_boundary, 32)
bcP  = fn.DirichletBC(Hh.sub(2), p_g, mesh_boundary, 31)
boundary_conditions = [bcU, bcP]

# initial conditions - poroelasticity
u_old   = fn.interpolate(fn.Constant((0.0, 0.0)), Vhf)
phi_old = fn.interpolate(fn.Constant(0.0), Zhf)
p_old   = fn.interpolate(fn.Constant(0.0), Qhf)

# initial conditions - reaction-diffusion
E_old = fn.interpolate(fn.Constant(0.0),Mhf)
n_old = fn.interpolate(fn.Constant(0.0),Mhf)

stim_t1   = 1.0
stim_t2   = 320.0
stim_dur1 = 3.0
stim_dur2 = 3.0
stim_amp  = 3.0

waveS1 = fn.Expression("amp*(x[0]<=0.01*width - width/2)",
                    amp=stim_amp, width=mesh_width, degree=2)
waveS2 = fn.Expression("amp*(x[1] < 0.5*height && x[0] < 0)",
                    amp = stim_amp, height=mesh_height, degree=2)

# reaction-diffusion stimulation term
def Istim(t):
    if (stim_t1 <= t and t <= stim_t1 + stim_dur1):
        return waveS1
    if (stim_t2 <= t and t <= stim_t2 + stim_dur2):
        return waveS2
    else:
        return fn.Constant(0.0)

# weak forms - poroelasticity
# left-hand side
pe_left_hand_side = \
        (c0/alpha + 1.0/lmbda) * (p/dt) * q * fn.dx - \
        (1.0/lmbda) * (phi/dt) * q * fn.dx + \
        (kappa/(eta * alpha)) * fn.dot(fn.grad(p), fn.grad(q)) * fn.dx - \
        (1.0/lmbda) * phi * psi * fn.dx + \
        (1.0/lmbda) * p * psi * fn.dx - \
        fn.div(u) * psi * fn.dx + \
        2.0 * mu * fn.inner(strain(u), strain(v)) * fn.dx - \
        phi * fn.div(v) * fn.dx

# right-hand side
pe_right_hand_side = \
        (c0/alpha + 1.0/lmbda) * (p_old/dt) * q * fn.dx - \
        (1.0/lmbda) * (phi_old/dt) * q * fn.dx + \
        fn.dot(f, v) * fn.dx + \
        fn.dot(v, c_1 * fn.outer(f_0, f_0) * fn.grad(n_old)) * fn.dx
        
# solutions
pe_solution = fn.Function(Hh)
rd_solution = fn.Function(Nh)

z     = fn.Function(Vhf)
sigma = fn.Function(Whf)

# reaction-diffusion left-hand side
rd_left_hand_side = \
        fn.dot(D(E_old, sigma) * fn.grad(E), fn.grad(F)) * fn.dx + \
        (n/dt) * m * fn.dx + \
        fn.dot(z, fn.grad(E)) * F * fn.dx + \
        fn.dot(z, fn.grad(n)) * m * fn.dx + \
        (E/dt) * F * fn.dx
            
# reaction-diffusion right-hand side
rd_right_hand_side = \
        (E_old/dt) * F * fn.dx + \
        RD_f(E_old, n_old) * F * fn.dx + \
        RD_g(E_old, n_old) * m * fn.dx + \
        (n_old/dt) * m * fn.dx

# time loop
iteration = 0
while t <= TFinal:
    # log time
    print "t =", t
    
    # solve poroelasticity
    fn.solve(pe_left_hand_side == pe_right_hand_side,
             pe_solution,
             boundary_conditions)
    
    # split
    u, phi, p = pe_solution.split()
    
    # calculate sigma and z
    sigma = fn.project(2 * mu * strain(u_old) - phi * fn.Identity(2) + \
            n_old * c_1 * fn.outer(f_0, f_0), Whf)
    z = fn.project((u - u_old)/dt, Vhf)
    
    rd_rhs_2 = rd_right_hand_side + Istim(t) * F * fn.dx
    
    # solve reaction-diffusion
    fn.solve(rd_left_hand_side == rd_rhs_2,
             rd_solution)
    
    # split
    E, n = rd_solution.split()
    
    # save if divisible by frequency
    if iteration % frequency == 0:
        u.rename("u", "u")
        phi.rename("phi", "phi")
        p.rename("p", "p")
        E.rename("E", "E")
        n.rename("n", "n")
        pvdU << (u, t)
        pvdPHI << (phi, t)
        pvdP << (p, t)
        pvdE << (E, t)
        pvdN << (n, t)
    
    # update solutions
    fn.assign(u_old, u)
    fn.assign(phi_old, phi)
    fn.assign(p_old, p)
    fn.assign(E_old, E)
    fn.assign(n_old, n)
    
    # update time
    t += dt
    iteration += 1
