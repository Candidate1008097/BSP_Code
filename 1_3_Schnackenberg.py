import fenics as fn
# set parameters
fn.parameters["form_compiler"]["representation"] = "uflacs"
fn.parameters["form_compiler"]["cpp_optimize"] = True

# mesh setup
nps   = 64  # number of points 
mesh  = fn.UnitSquareMesh(nps,nps)
fileu = fn.File(mesh.mpi_comm(), "Output/1_3_Schnackenberg/u.pvd")
filev = fn.File(mesh.mpi_comm(), "Output/1_3_Schnackenberg/v.pvd")

# function spaces
P1  = fn.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
Mh  = fn.FunctionSpace(mesh, "Lagrange", 1) 
Nh  = fn.FunctionSpace(mesh, fn.MixedElement([P1,P1]))
Sol = fn.Function(Nh)

# trial and test functions
u,  v  = fn.TrialFunctions(Nh)
uT, vT = fn.TestFunctions(Nh)

# model constants
a  = fn.Constant(0.1305)
b  = fn.Constant(0.7695)
c1 = fn.Constant(0.05)
c2 = fn.Constant(1.0)
d  = fn.Constant(170.0)

# initial value
uinit = fn.Expression('a + b + 0.001 * exp(-100.0 * (pow(x[0] - 1.0/3, 2)' +
	                  ' + pow(x[1] - 0.5, 2)))', a=a, b=b, degree=3)

# old results
uold = fn.interpolate(uinit, Mh)
vold = fn.interpolate(fn.Constant(b * pow(a + b, -2)), Mh)

# computation parameters
hsize = 1.0/nps; 
t = 0.0; CFL = 5.0; dt = CFL * hsize * hsize; T = 2.0; frequencySave = 100;

# weak form
Left  = u/dt * uT * fn.dx + v/dt * vT * fn.dx \
        + c1 * fn.inner(fn.grad(u), fn.grad(uT)) * fn.dx \
        + c2 * fn.inner(fn.grad(v), fn.grad(vT)) * fn.dx

Right = uold * uT/dt * fn.dx + vold * vT/dt * fn.dx \
        + d * (a - uold + uold * uold * vold) * uT * fn.dx \
        + d * (b - uold * uold * vold) * vT * fn.dx

AA = fn.assemble(Left)
solver = fn.LUSolver(AA)
solver.parameters["reuse_factorization"] = True

# time loop
inc = 0

while (t <= T):

    print "t =", t
    
    # solve
    BB = fn.assemble(Right)
    solver.solve(Sol.vector(), BB)
    
    # output to file
    u,v = Sol.split()
    if (inc % frequencySave == 0):
        u.rename("u","u"); fileu << (u,t)
        v.rename("v","v"); filev << (v,t)
        
    # update old values
    fn.assign(uold,u); fn.assign(vold,v)
    
    # increment
    t += dt; inc += 1
