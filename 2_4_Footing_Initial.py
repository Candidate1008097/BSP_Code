import fenics as fn
fn.parameters["form_compiler"]["representation"] = "uflacs"
fn.parameters["form_compiler"]["cpp_optimize"] = True

# Create mesh and define function space
L = 50.0

mesh = fn.RectangleMesh(fn.Point(-50,0), fn.Point(50,75),50,30)
nn = fn.FacetNormal(mesh) 


# Defintion of function spaces
Vh = fn.VectorElement("CG", mesh.ufl_cell(), 2)
Zh = fn.FiniteElement("CG", mesh.ufl_cell(), 1)
Qh = fn.FiniteElement("CG", mesh.ufl_cell(), 2)


# spaces for displacement and total pressure should be compatible
# whereas the space for fluid pressure can be "anything".
# In particular the one for total pressure
Hh = fn.FunctionSpace(mesh, fn.MixedElement([Vh,Zh,Qh]))

# trial and test functions
u, phi, p = fn.TrialFunctions(Hh)
v, psi, q = fn.TestFunctions(Hh)

fileU   = fn.File(mesh.mpi_comm(), "Output/2_4_Footing_initial/u.pvd")
filePHI = fn.File(mesh.mpi_comm(), "Output/2_4_Footing_initial/phi.pvd")
fileP   = fn.File(mesh.mpi_comm(), "Output/2_4_Footing_initial/p.pvd")

# model constants
E     = fn.Constant(3.0e4)
nu    = fn.Constant(0.4995)
lmbda = E*nu/((1.+nu)*(1.-2.*nu)) 
mu    = E/(2.*(1.+nu))

c0    = fn.Constant(1.0e-3)
kappa = fn.Constant(1.0e-4)
alpha = fn.Constant(0.1)
sigma0= fn.Constant(1.5e4)
eta   = fn.Constant(1.0)

# functions
# body force
f = fn.Constant((0.0,0.0))

# dirichlet data
u_g = fn.Constant((0.0,0.0))
p_g = fn.Constant(0.0)

h_g = fn.Constant((0.0,-sigma0))

def strain(v): 
    return fn.sym(fn.grad(v))

# boundary conditions
bdry = fn.MeshFunction("size_t", mesh, 1)

bdry.set_all(0)

# foot boundary
FooT = fn.CompiledSubDomain("x[0]>= -0.4*L && x[0] <= 0.4*L && near(x[1],75.0) && on_boundary", L=L)
GammaU = fn.CompiledSubDomain("(near(x[0],-50.0) || near(x[1],0.0) || near(x[0],50.0) ) && on_boundary")
GammaU.mark(bdry,31); FooT.mark(bdry,32);
ds = fn.Measure("ds", subdomain_data=bdry)

bcU = fn.DirichletBC(Hh.sub(0), u_g, bdry, 31)
bcP = fn.DirichletBC(Hh.sub(2), p_g, bdry, 32)
bcs = [bcU,bcP]

# weak forms
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

# save solutions
u, phi, p = Sol.split()
u.rename("u","u"); fileU << u
p.rename("p","p"); fileP << p
phi.rename("phi","phi"); filePHI << phi
