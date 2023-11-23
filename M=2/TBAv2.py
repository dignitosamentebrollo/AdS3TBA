# Solver for TBA Eqs of tensionless AdS3
#################
import cmath as c
from scipy import integrate
from scipy import fft
from scipy import optimize
from numpy import *
# -----------------

#-----------------
def exp2(x,A=500,vec=True): # exp function that avoids overflows
	if vec==False: #the exp function is from numpy, so I have to pass vectors
		x=[x] 
	N=len(x)
	res=zeros(N,dtype=complex)
	for j in range(N):
		if abs(x[j]) >= A: # Note: exp(-500)=7.12e-218~0; exp(500)=1.40e+217~ +infty << our approximation
			res[j]= exp(sign(x[j])*A)
		else:
			res[j]= exp(x[j])
	if vec==False:
		return res[0].real
	return res.real

def log2(x,A=500,vec=True): # log function that avoids overflows
	if vec==False:	#the log function is from numpy, so I have to pass vectors
		x=[x] 
	N=len(x)
	res=zeros(N,dtype=complex)
	for j in range(N):
		if abs(x[j]) >= exp(-A): # Note: exp(-500)=7.12e-218~0; exp(500)=1.40e+217~ +infty << our approximation
			res[j]= log(complex(x[j]))
		elif x[j]==0:
			res[j]= -A
		else:
			res[j]= log(complex(sign(x[j])*exp(-A)))
	if vec==False:
		return res[0]
	return res

def sinh2(x,A=400,B=exp(-250),vec=True): # sinh function that avoids overflows
	if vec==False: #the sinh function is from numpy, so I have to pass vectors
		x=[x] 
	N=len(x)
	res=zeros(N,dtype=complex)
	for j in range(N):
		if abs(x[j]) >= A: # Note: sinh(400)=2.61e+163~ +infty << our approximation
			res[j]= sinh(sign(x[j])*A)
		elif x[j]==0:
			res[j]= B
		elif abs(x[j]) <= B: # Note: sinh(B)~B
			res[j]= sign(x[j])*B
		else:
			res[j]= sinh(x[j])
	if vec==False:
		return res[0]
	return res

from scipy.special import spence
def varphi(x,vec=False): #smatrix dressing factor
    if vec==False:
        x=complex(x)
    return (1j/pi)*spence(1+exp(-x))-(1j/pi)*spence(1-exp(-x))+(1j*x/pi)*log2(1-exp(-x),vec=vec)-(1j*x/pi)*log2(1+exp(-x),vec=vec)+(1j*pi/4)

#----------------	
def shift(vec): # it assumes N even # Note: twice a shift == original order
	N=len(vec)
	n=int(N//2)
	new_vec=zeros(N,dtype=complex)
	for j in range(n):
		new_vec[j]=vec[j+n]
		new_vec[j+n]=vec[j]
	return new_vec
#-------------------
# The following function compute the conv. defined as \int dk' f(k-k')g(k')
# In the paper the convs. as defined as \int dk' f(k'-k)g(k')
# So in the next part of the code the kernel K(k) are defined as K(-k)
from scipy.fft import fft,ifft
def fast_conv(g,K,Lambda,N):
	fK=fft(K)
	fg=fft(g)
	conv=ifft(fK*fg)
	return shift(real(conv))*(2*Lambda/(N)) 


# The following function compute the conv. for the exact Bethe eqns
# We have to cure the case 1/sinh(0) which is bad defined
# but can be extend continuosly the integrand as explained in the paper
def int_conv(x0, gamma2, g, kset, corr, lim):
    def der(gamma1, gamma2, ind):
        # gamma1 is the one in which I compute the derivative
        # derivative of Y0 in the pole
        return exp(corr[ind])*0.5*tanh(-gamma1)*(-1)*tanh((-gamma2-gamma1)/2)*tanh((gamma2-gamma1)/2)
    div = False
    for i, x in enumerate(kset):
        if abs(x0-x) <= exp(-250):
            div = True
            ind = i
            print('Wow')
    integrand = g*sstar(x0-kset)
    if div == True:
        integrand[ind] = der(x0, gamma2, ind)*lim
    conv = integrate.simps(integrand, kset)
    return conv

# ---------------------
###############################
# Kernels and S-matrix
def S0(x):
	return -1j*tanh(x/2-(pi/4)*1j)

def logS0(x): #without dressing factor (for exact equations)
    return log2(-1j*tanh(x/2-(pi/4)*1j),vec=False)

def logS(x): #with dressing factor (for the asymptotics Bethe equations)
    return log2(-1j*tanh(x/2-(pi/4)*1j),vec=False) + 2*varphi(x)

def logSstar(x):
	return log2(-1j*tanh(x/2))

def s(x):
	return 1/(2*pi*cosh(x))

def sstar(x): #sign inverted as commented before
	return -1/(2j*pi*sinh2(x))
#-----------------------------
###############################
# Energies and momenta
def pmirr(x): #divided by h
	return -2/(sinh2(x))

def E(x): #divided by h
	return 2/(cosh(x))

def Emirr(x):
	return -log2(((1-exp2(x))/(1+exp2(x)))**2)

def p(x):
	return -1j*log2(((exp2(x,vec=False)-1j)/(exp2(x,vec=False)+1j))**2,vec=False)

###############################
def error(a,b):
	N=len(a)
	#---------------
	# Abs error 
	abs_err=max([abs(a[j]-b[j]) for j in range(N)])
	#---------------

	return abs_err
#-------------------------------



# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Thermodynamic-Bethe-Ansatz Equations
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def exact_bethe(L, f0, f, n1, n2, gamma1, gamma2, corr, Lambda, N,N0,x0=[0.5,0.8]):
    kset = arange(-Lambda, Lambda, (2*Lambda)/N)

    def f_pol(x):
        x1 = x[0]
        x2 = x[1]
        z1 = -1j*L*p(x1) - N0*int_conv(gamma1, gamma2, log2(abs(1+f0)), kset, corr, 1/(1j*pi)) - 4*int_conv(
            gamma1, gamma2, log2(abs(1-f)), kset, corr, (-2)/(1j*pi)) - 2j*pi*n1 + logS0(-2*x1) + logS0(x2-x1) + logS0(-x2-x1)
        z2 = -1j*L*p(x2) - N0*int_conv(gamma2, gamma1, log2(abs(1+f0)), kset, corr, 1/(1j*pi)) - 4*int_conv(
            gamma2, gamma1, log2(abs(1-f)), kset, corr, (-2)/(1j*pi)) - 2j*pi*n2 + + logS0(-2*x2) + logS0(x1-x2) + logS0(-x1-x2)
        return [z1.imag, z2.imag]
    def pol_eqn(x): return f_pol(x)
    sol = optimize.root(pol_eqn,x0,tol=1e-12)
    if sol.success == True: #return the solution only in case of convergence
        return sol.x
    else:
        return sol.success


def asymptotic_bethe(L, n1, n2, x0=[2.1,2]):
    def f_pol(x):
        x1 = x[0]
        x2 = x[1]
        z1 = -1j*L*p(x1) + logS(-2*x1) + logS(x2-x1) + logS(-x2-x1) - 2j*pi*n1 
        z2 = -1j*L*p(x2) + logS(-2*x2) + logS(x1-x2) + logS(-x1-x2) - 2j*pi*n2 
        return [z1.imag, z2.imag]
    def pol_eqn(x): return f_pol(x)
    sol = optimize.root(pol_eqn, x0, tol=1e-10)
    if sol.success == True: #return the solution only in case of convergence
        return sol.x
    else:
        return sol.success
    

#THE ASYMPTOTIC FOR Y AND Y0 IS TRIVIAL IN THIS CASE


def tba_solver(L, n1, n2, N0, Lambda=50, N=int(2**10), prec=1e-13, min_prec=1e-10, iterations=int(1e03), damping_param=0.6,x0=[0.5,0.8]):
    # ------------------------
    kset = arange(-Lambda, Lambda, (2*Lambda)/N)
    # ------------------------
    
    
    # -------------------
    # 	INITIALIZATION
    # -------------------

    f0 = zeros(N)  # initialize =>> Y0 asymptotics
    [gamma1, gamma2] = asymptotic_bethe(L, n1, n2,x0)  # initialize =>> asymptotic bethe roots
    # ------------------
    # -- Driving term
    a_1 = -L*Emirr(kset)

    # ------------------
    # -- Integral Kernels dicretized

    s_set = s(kset)
    ##############################
    ##############################

    # -------------------
    # -- Starting output
    print('Mode \u03BD1,\u03BD3=', n1,n2)
    print('Asymptotic Bethe Roots:')
    print(gamma1, gamma2)
    print('Asymptotic Energy:',4/cosh(gamma1)+4/cosh(gamma2))
    
    asy_gamma1=gamma1
    asy_gamma2=gamma2
    asy_en=4/cosh(gamma1)+4/cosh(gamma2)

    # -------------------
    # 		RUN
    # -------------------
    err = 1
    cont = 0

    while err > prec and cont < iterations:
        conv1 = fast_conv(log2((1+f0)**N0), s_set, Lambda, N)
        f = exp2(conv1)*exp2(logSstar(-gamma1-kset)+logSstar(gamma1-kset)+logSstar(-gamma2 -kset)+logSstar(gamma2 -kset))
    # ---------------------------
        conv2 = 4*fast_conv(log2(abs((1 - f))), s_set, Lambda, N)
        corr = a_1+conv1+conv2
        f0_tmp = exp(corr)*exp2(logSstar(-gamma1-kset)+logSstar(gamma1-kset)+logSstar(-gamma2 -kset)+logSstar(gamma2 -kset))
    # ---------------------------
        [gamma1_tmp, gamma2_tmp] = exact_bethe(L, f0_tmp, f, n1, n2, gamma1, gamma2, corr, Lambda, N,N0,x0)


    # get the error
        err=max(error(f0_tmp,f0),abs(gamma1-gamma1_tmp)*0.001,abs(gamma2-gamma2_tmp)*0.001)

    # update
        f0 = (1-damping_param)*f0_tmp+damping_param*f0
        gamma1 = gamma1_tmp
        gamma2 = gamma2_tmp
        cont += 1
    # -----------------------------

    if cont == iterations:
        print('Max iterations reached')
        if err > min_prec:
            print('Failed convergence')
            f0 = zeros(N)
            f = zeros(N)
            gamma1 = 0
            gamma2 = 0
            # quit()
    # ----------------------------
    # -- Final output
    print('The final gamma are:')
    print(gamma1,gamma2)
    print('The energy in this state is', energy(
        L, f0, gamma1, gamma2, Lambda, N,N0))

    return f0, f, gamma1, gamma2, asy_gamma1,asy_gamma2,asy_en


def asy_tba_solver(L, n1, n2, Lambda=50, N=int(2**10)):
    # ------------------------
    kset = arange(-Lambda, Lambda, (2*Lambda)/N)
    # ------------------------

    # -------------------
    # 	INITIALIZATION
    # -------------------

    [gamma1, gamma2] = asymptotic_bethe(
        L, n1, n2)  # initialize =>> asymptotic bethe roots

    # ------------------
    # -- Driving term
    a_1 = -L*Emirr(kset)

    # ------------------
    ##############################
    ##############################

    # -------------------
    # print('n=', n)
    # print('Asymptotic gamma:',gamma)
    # ---------------------------

    return gamma1, gamma2


def energy(L, f0, gamma1, gamma2, Lambda, N, N0):
    kset = arange(-Lambda, Lambda, (2*Lambda)/N)
    dp = [2*cosh(j)/(sinh2(j, vec=False)**2) for j in kset]
    dp[N//2] = 0  # in the pole the integrand is zero
    # no asymptotic problems due to exp suppression
    integ = -(N0/(2*pi))*integrate.simps(dp*(log2(1+f0)), kset)
    return (integ + 4/(cosh(gamma1))+4/(cosh(gamma2))).real
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
