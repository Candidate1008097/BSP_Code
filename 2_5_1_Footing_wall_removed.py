import fenics as fn
fn.parameters["form_compiler"]["representation"] = "uflacs"
fn.parameters["form_compiler"]["cpp_optimize"] = True

# Create mesh and define function space
size = width, height = 1.0, 0.75

mesh = fn.RectangleMesh(fn.Point(-width/2, 0.0), fn.Point(width/2, height),
                        52, 39) # 52 * 0.75 = 39, elements are square
nn = fn.FacetNormal(mesh) 


# Defintion of function spaces
Vh = fn.VectorElement("CG", mesh.ufl_cell(), 2)
Zh = fn.FiniteElement("CG", mesh.ufl_cell(), 1)
Qh = fn.FiniteElement("CG", mesh.ufl_cell(), 2)


# spaces for displacement and total pressure should be compatible
# whereas the space for fluid pressure can be "anything". In particular the one for total pressure

Hh = fn.FunctionSpace(mesh, fn.MixedElement([Vh,Zh,Qh]))

(u, phi, p) = fn.TrialFunctions(Hh)
(v, psi, q) = fn.TestFunctions(Hh)

fileU   = fn.File(mesh.mpi_comm(), "Output/2_5_1_Footing_wall_removed/u.pvd")
filePHI = fn.File(mesh.mpi_comm(), "Output/2_5_1_Footing_wall_removed/phi.pvd")
fileP   = fn.File(mesh.mpi_comm(), "Output/2_5_1_Footing_wall_removed/p.pvd")

# ******** Model constants ********** #
E     = 3.0e4
nu    = 0.4995
lmbda = E*nu/((1.+nu)*(1.-2.*nu)) 
mu    = E/(2.*(1.+nu))

c0    = 1.0e-3
kappa = 1.0e-6
alpha = 0.1
sigma0= 1.5e4
eta   = fn.Constant(1.0)

# ******** Functions ********** #

# body force
f = fn.Constant((0.0,0.0))

# Dirichlet data
u_g = fn.Constant((0.0,0.0))
p_g = fn.Constant(0.0)

h_g = fn.Constant((0.0,-sigma0))

def strain(v): 
    return fn.sym(fn.grad(v))

# boundary conditions
bdry = fn.MeshFunction("size_t", mesh, 1)

bdry.set_all(0)
FooT = fn.CompiledSubDomain(
        "(x[0] >= -0.2) && (x[0] <= 0.2) && near(x[1], 0.75) && on_boundary")
GammaU = fn.CompiledSubDomain(
        "( near(x[0], -0.5) || near(x[1], 0.0) ) && on_boundary")
GammaU.mark(bdry,31); FooT.mark(bdry,32);
ds = fn.Measure("ds", subdomain_data=bdry)

bcU = fn.DirichletBC(Hh.sub(0), u_g, bdry, 31)
bcP = fn.DirichletBC(Hh.sub(2), p_g, bdry, 32)
bcs = [bcU,bcP]

# ********  Weak forms ********** #

PLeft =  2*mu*fn.inner(strain(u),strain(v)) * fn.dx \
         - fn.div(v) * phi * fn.dx \
         + (c0/alpha + 1.0/lmbda)* p * q * fn.dx \
         + kappa/(alpha*nu) * fn.dot(fn.grad(p),fn.grad(q)) * fn.dx \
         - 1.0/lmbda * phi * q * fn.dx \
         - fn.div(u) * psi * fn.dx \
         + 1.0/lmbda * psi * p * fn.dx \
         - 1.0/lmbda * phi * psi * fn.dx


PRight = fn.dot(f,v) * fn.dx + fn.dot(h_g,v)*ds(32)


Sol = fn.Function(Hh)

# solve
fn.solve(PLeft == PRight, Sol, bcs)

u,phi,p = Sol.split()
u.rename("u","u"); fileU << u
p.rename("p","p"); fileP << p
phi.rename("phi","phi"); filePHI << phi
