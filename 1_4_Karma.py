import fenics as fn
import time
# fenics parameters
fn.parameters["form_compiler"]["representation"] = "uflacs"
fn.parameters["form_compiler"]["cpp_optimize"] = True

# file outputs
fileE = fn.File("Output/1_4_Karma/E.pvd")
filen = fn.File("Output/1_4_Karma/n.pvd")
t = 0.0; dt = 0.3; Tfinal = 600.0; frequency = 100;

# mesh
L = 6.72; nps = 64;
mesh = fn.RectangleMesh(fn.Point(0, 0), fn.Point(L, L),
			            nps, nps, "crossed")

# element and function spaces
Mhe = fn.FiniteElement("CG", mesh.ufl_cell(), 1)
Mh  = fn.FunctionSpace(mesh, Mhe)
Nh  = fn.FunctionSpace(mesh, fn.MixedElement([Mhe,Mhe]))

# trial and test functions
v, n   = fn.TrialFunctions(Nh)
w, m   = fn.TestFunctions(Nh)

# solution
Ksol   = fn.Function(Nh)

# model constants
diffScale = fn.Constant(1e-3)
D0 = 1.1 * diffScale
tauv  = fn.Constant(2.5)
taun  = fn.Constant(250.0)
Re    = fn.Constant(1.0)
M     = fn.Constant(5.0)
beta  = fn.Constant(0.008)
vstar = fn.Constant(1.5415)
vh    = fn.Constant(3.0)
vn    = fn.Constant(1.0)

# functions
def RR(n):
    return (1.0 - (1.0 - fn.exp(-Re)) * n ) / (1.0 - fn.exp(-Re))

def HH(r):
    return 0.5 * (fn.sign(r) + 1.0)

def g(v,n):
    return 1.0/taun * (RR(n) * HH(v-vn) - (1.0 - HH(v-vn))*n)

def f(v,n):
    return 1.0/tauv * (-v + (vstar - pow(n, M)) \
    	   * (1.0 - fn.tanh(v - vh)) * pow(v, 2) * 0.5)

# initial conditions and stimulation function
vold = fn.interpolate(fn.Constant(0.0), Mh)
nold = fn.interpolate(fn.Constant(0.0), Mh)

stim_t1   = 1.0
stim_t2   = 350.0
stim_dur1 = 3.0
stim_dur2 = 3.0
stim_amp  = 3.0

waveS1 = fn.Expression("amp * (x[0] <= 0.01 * L)",
	                   amp=stim_amp, L=L, degree=2)
waveS2 = fn.Expression("amp * (x[1] < 0.5 * L && x[0] < 0.5 * L)",
	                   amp=stim_amp, L=L, degree=2)


def Istim(t):
    if (stim_t1 <= t and t <= stim_t1 + stim_dur1):
        return waveS1
    if (stim_t2 <= t and t <= stim_t2 + stim_dur2):
        return waveS2
    else:
        return fn.Constant(0.0)
    
# weak form
WeakKarma = v/dt * w * fn.dx + fn.inner(D0 * fn.grad(v), fn.grad(w)) * fn.dx \
            + n/dt * m * fn.dx 

Reactions = vold/dt * w * fn.dx + f(vold, nold) * w * fn.dx \
            + nold/dt * m * dx + g(vold, nold) * m * fn.dx
    
# ************* Time loop ********
start = time.clock(); inc = 0

while (t <= Tfinal):
    
    print "t =", t
    
    # solve
    KR = Reactions + Istim(t)*w*dx 
    fn.solve(WeakKarma == KR, Ksol,
    	solver_parameters={'linear_solver':'bicgstab'})
    v, n = Ksol.split()
    
    # save solution
    if (inc % frequency == 0):
        v.rename("v","v"); filev << (v,t)
        n.rename("n","n"); filen << (n,t)
         
    # update
    fn.assign(vold, v); fn.assign(nold, n)
    
    # increment
    t += dt; inc += 1

# time elapsed   
print "Time elapsed: ", time.clock()-start
