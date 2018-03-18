# import
import fenics as fn

# set parameters
fn.parameters["form_compiler"]["representation"] = "uflacs"
fn.parameters["form_compiler"]["cpp_optimize"] = True

# create mesh and function spaces
L = 50.0
mesh = fn.BoxMesh(
        fn.Point(-L, 0.0, -L),
        fn.Point(L, 2*L, L),
        20, 5, 20)
nn = fn.FacetNormal(mesh)

Vh = fn.VectorElement("CG", mesh.ufl_cell(), 2)
Zh = fn.FiniteElement("CG", mesh.ufl_cell(), 1)
Qh = fn.FiniteElement("CG", mesh.ufl_cell(), 2)

Hh = fn.FunctionSpace(mesh, fn.MixedElement([Vh, Zh, Qh]))

# variables to solve for, and test functions
u, phi, p = fn.TrialFunctions(Hh)
v, psi, q = fn.TestFunctions(Hh)

# file output
fileU   = fn.File(mesh.mpi_comm(), "Output/2_5_2_Footing_3D/u.pvd")
filePHI = fn.File(mesh.mpi_comm(), "Output/2_5_2_Footing_3D/phi.pvd")
fileP   = fn.File(mesh.mpi_comm(), "Output/2_5_2_Footing_3D/p.pvd")

# constants
#E       = fn.Constant(3.0e4)
#nu      = fn.Constant(0.4995)
#lmbda   = fn.Constant(E*nu/((1.+nu)*(1.-2.*nu)))
#mu      = fn.Constant(E/(2.*(1.+nu)))
#
#c_0     = fn.Constant(1.0e-3)
#kappa   = fn.Constant(1.0e-4)
#alpha   = fn.Constant(0.1)
#sigma_0 = fn.Constant(1.5e4)
#eta     = fn.Constant(1.0)
E       = 3.0e4
nu      = 0.4995
lmbda   = E*nu/((1.+nu)*(1.-2.*nu))
mu      = E/(2.*(1.+nu))

c_0     = 1.0e-3
kappa   = 1.0e-4
alpha   = 0.1
sigma_0 = 6e4
eta     = 1.0

# functions
f = fn.Constant((0.0, 0.0, 0.0))

# dirichlet data
u_g = fn.Constant((0.0, 0.0, 0.0))
p_g = fn.Constant(0.0)

h_g = fn.Constant((0.0, -sigma_0, 0.0))

# symmetric strain tensor
def strain(v):
    return fn.sym(fn.grad(v))

# boundary and boundary conditions
bdry = fn.MeshFunctionSizet(mesh, 2)

bdry.set_all(0)
FooT = fn.CompiledSubDomain(
        "sqrt(x[0]*x[0] + x[2]*x[2]) >= -0.4*L &&" +
        "sqrt(x[0]*x[0] + x[2]*x[2]) <=  0.4*L &&" +
        "near(x[1], 2*L) && on_boundary", L=L)
GammaU = fn.CompiledSubDomain(
        "(near(x[0], -L) || near(x[0], L) ||" +
        " near(x[2], -L) || near(x[2], L) ||" +
        " near(x[1], 0.0)) && on_boundary", L=L)
GammaU.mark(bdry, 31)
FooT.mark(bdry, 32)
fn.ds = fn.Measure("ds", subdomain_data=bdry)

bcU = fn.DirichletBC(Hh.sub(0), u_g, bdry, 31)
bcP = fn.DirichletBC(Hh.sub(2), p_g, bdry, 32)
bcs = [bcU, bcP]

# weak forms
PLeft = 2*mu*fn.inner(strain(u), strain(v)) * fn.dx \
        - fn.div(v) * phi * fn.dx \
        + (c_0/alpha + 1.0/lmbda) * p * q * fn.dx \
        + kappa/(alpha*nu) * fn.dot(fn.grad(p), fn.grad(q)) * fn.dx \
        - 1.0/lmbda * phi * q * fn.dx \
        - fn.div(u) * psi * fn.dx \
        + 1.0/lmbda * psi * p * fn.dx \
        - 1.0/lmbda * phi * psi * fn.dx

PRight = fn.dot(f, v) * fn.dx + fn.dot(h_g, v)*fn.ds(32)

Sol = fn.Function(Hh)

# solve!
fn.solve(PLeft == PRight, Sol, bcs)

u, phi, p = Sol.split()

# save
u.rename("u", "u")
fileU << u

p.rename("p", "p")
fileP << p

phi.rename("phi", "phi")
filePHI << phi
