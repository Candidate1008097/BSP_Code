import fenics as fn
fn.parameters["form_compiler"]["representation"] = "uflacs"
fn.parameters["form_compiler"]["cpp_optimize"] = True

# create mesh
mesh_size = mesh_width, mesh_height = 1.0, 0.75
foot_width = 0.4
mesh = fn.RectangleMesh(fn.Point(-mesh_width/2, 0.0),
                        fn.Point( mesh_width/2, mesh_height),
                        52, 39)  # square elements

# time-related variables
t = 0.0
dt = 0.1
TFinal = 10 * dt
frequency = 1

# normals
nn = fn.FacetNormal(mesh)

# element spaces
Vh = fn.VectorElement("CG", mesh.ufl_cell(), 2)
Zh = fn.FiniteElement("CG", mesh.ufl_cell(), 1)
Qh = fn.FiniteElement("CG", mesh.ufl_cell(), 2)

# function spaces
Vhf = fn.FunctionSpace(mesh, Vh)
Zhf = fn.FunctionSpace(mesh, Zh)
Qhf = fn.FunctionSpace(mesh, Qh)
Hh = fn.FunctionSpace(mesh, fn.MixedElement([Vh,Zh,Qh]))

# functions and test functions
u, phi, p = fn.TrialFunctions(Hh)
v, psi, q = fn.TestFunctions(Hh)

# pvd files
pvdU   = fn.File(mesh.mpi_comm(), "Output/2_6_Footing_time_dependent/u.pvd")
pvdPHI = fn.File(mesh.mpi_comm(), "Output/2_6_Footing_time_dependent/phi.pvd")
pvdP   = fn.File(mesh.mpi_comm(), "Output/2_6_Footing_time_dependent/p.pvd")

# constants for model
E      = fn.Constant(3.0e4)
nu     = fn.Constant(0.4995)
lmbda  = E*nu/((1.+nu)*(1.-2.*nu)) 
mu     = E/(2.*(1.+nu))

c0     = fn.Constant(1.0e-3)
kappa  = fn.Constant(1.0e-4)
alpha  = fn.Constant(0.1)
sigma0 = fn.Constant(1.5e4)
eta    = fn.Constant(1.0)

# functions
# body force
f = fn.Constant((0.0, 0.0))  # no body force (gravity)

# dirichlet boundary conditions
u_g = fn.Constant((0.0, 0.0))  # 0 displacement on walls
p_g = fn.Constant(0.0)  # 0 pressure on free surface
h_g = fn.Expression(("0","-sigma0*t"), sigma0=sigma0, t=0.0, degree=3)

# symmetric strain tensor
def strain(u):
    return fn.sym(fn.grad(u))

# boundaries
mesh_boundary = fn.MeshFunctionSizet(mesh, 1)
mesh_boundary.set_all(0)

# section of boundary on which foot presses
foot_boundary = fn.CompiledSubDomain(
        "(x[0] >= -foot_width/2) && (x[0] <= foot_width/2)" +
        "&& near(x[1], mesh_height) && on_boundary",
        foot_width=foot_width, mesh_height=mesh_height)

pressure_boundary = fn.CompiledSubDomain(
        "((x[0] < -foot_width/2) || (x[0] > foot_width/2))" +
        "&& near(x[1], mesh_height) && on_boundary",
        foot_width=foot_width, mesh_height=mesh_height)

# section of boundary that is a rigid wall
wall_boundary = fn.CompiledSubDomain(
        "(near(x[0], -mesh_width/2) || near(x[0], mesh_width/2)" +
        "|| near(x[1], 0.0)) && on_boundary",
        mesh_width=mesh_width)

# mark boundaries
foot_boundary.mark(mesh_boundary, 31)
wall_boundary.mark(mesh_boundary, 32)
pressure_boundary.mark(mesh_boundary, 33)

# ds, for integrating
ds = fn.Measure("ds", subdomain_data=mesh_boundary)

# boundary conditions
bcU  = fn.DirichletBC(Hh.sub(0), u_g, mesh_boundary, 32)
bcP  = fn.DirichletBC(Hh.sub(2), p_g, mesh_boundary, 33)
boundary_conditions = [bcU, bcP]

# initial conditions
#u_old   = fn.interpolate(fn.Constant((0.0, 0.0)), Vhf)
phi_old = fn.interpolate(fn.Constant(0.0), Zhf)
p_old   = fn.interpolate(fn.Constant(0.0), Qhf)

# weak forms
# left-hand side
left_hand_side = \
        (c0/alpha + 1.0/lmbda) * (p/dt) * q * fn.dx - \
        (1.0/lmbda) * (phi/dt) * q * fn.dx + \
        (kappa/(eta * alpha)) * fn.dot(fn.grad(p), fn.grad(q)) * fn.dx - \
        (1.0/lmbda) * phi * psi * fn.dx + \
        (1.0/lmbda) * p * psi * fn.dx - \
        fn.div(u) * psi * fn.dx + \
        2.0 * mu * fn.inner(strain(u), strain(v)) * fn.dx - \
        phi * fn.div(v) * fn.dx

# right-hand side
right_hand_side = \
        (c0/alpha + 1.0/lmbda) * (p_old/dt) * q * fn.dx - \
        (1.0/lmbda) * (phi_old/dt) * q * fn.dx + \
        fn.dot(f, v) * fn.dx + \
        fn.dot(h_g, v) * ds(31)

# time loop
iteration = 0
while t <= TFinal:
    # log time
    print "t =", t
    
    # update force
    h_g.t = t
    
    # solution
    solution = fn.Function(Hh)
    
    # solve
    fn.solve(left_hand_side == right_hand_side,
             solution,
             boundary_conditions)
    
    # split
    u, phi, p = solution.split()
    
    # save if divisible by frequency
    if iteration % frequency == 0:
        u.rename("u", "u")
        phi.rename("phi", "phi")
        p.rename("p", "p")
        pvdU << (u, t)
        pvdPHI << (phi, t)
        pvdP << (p, t)
    
    # update solutions
    #fn.assign(u_old, u)
    fn.assign(phi_old, phi)
    fn.assign(p_old, p)
    
    # update time
    t += dt
    iteration += 1
