#Solver for TBA and Bethe roots Eqs of tensionless AdS3
#################
from numpy import *
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
from scipy import integrate
def int_conv(x0,g,kset,corr,A0,lim):
	def der(gamma,ind):
		return -A0*exp(corr[ind])*0.5*tanh(-gamma) #derivate of Y0 in the pole
	div=False
	for i,x in enumerate(kset):
		if abs(x0-x)<=exp(-250):
			div=True
			ind=i
			print('Wow')
	integrand=g*sstar(x0-kset)
	if div==True:
		integrand[ind]=der(x0,ind)*lim
	conv = integrate.simps(integrand,kset)
	return conv

#---------------------
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



#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Thermodynamic-Bethe-Ansatz Equations
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# The program is more stable if the integral convolutions
# of the exact Bethe are computed in the previous gamma
# of the iteration process (as commented in the paper)
from scipy import optimize
def exact_bethe(L,f0,f,n,gamma,A0,corr,Lambda,N,N0):
	kset=arange(-Lambda,Lambda,(2*Lambda)/N)
	def f_pol(x):
		z=-1j*L*p(x) - N0*int_conv(gamma,log2((1+f0)),kset,corr,A0,1/(1j*pi)) + logS0(-2*x) - 4*int_conv(gamma,log2((1-f)),kset,corr,A0,(-2)/(1j*pi)) - 2j*pi*n
		return z.imag #the equation is purely immaginary
	pol_eqn=lambda x: f_pol(x)
	sol=optimize.root_scalar(pol_eqn,bracket=[0,7.5],xtol=1e-14)
	return sol.root

def asymptotic_bethe(L,n):
	def f_pol(x):
		z=-1j*L*p(x) + logS(-2*x) - 2j*pi*n
		return z.imag #the equation is purely immaginary
	pol_eqn=lambda x: f_pol(x)
	sol=optimize.root_scalar(pol_eqn,bracket=[0,7.5],xtol=1e-14)
	return sol.root

# Large \gamma behaviour of the Y-functions
import cmath as c
def asymptotics(N0):
	def f(x):
		A=x[0]
		A0=x[1]
		return [abs(c.log(A)-(N0/2)*c.log(1+A0)-1j*pi), abs(-c.log(A0)+c.log(A)+c.log((1-A)**2))]
	return optimize.fsolve(f,x0=[-0.2,-0.4],xtol=1e-13)
	
def tba_solver(L,n,A,A0,N0,Lambda=50,N=int(2**10),prec=1e-14,min_prec=1e-10,iterations=int(1e03),damping_param=0.6):
	#------------------------
	kset=arange(-Lambda,Lambda,(2*Lambda)/N)
	#------------------------

	#-------------------
	# 	INITIALIZATION 
	#-------------------

	f0=A0*ones(N) # initialize =>> Y0 asymptotics
	gamma=asymptotic_bethe(L,n)    #initialize =>> asymptotic bethe roots

	#------------------
	#-- Driving term
	a_1 = -L*Emirr(kset)

	#------------------
	#-- Integral Kernels dicretized
	s_set=s(kset)

	#-------------------
	#-- Starting output
	print('Mode \u03BD=', n)
	#print('Asymptotic gamma:',gamma)


	#-------------------
	# 		RUN 
	#-------------------
	err=1
	cont=0

	gamma_plot=[gamma] #list for the plot of gamma convergence (not useful for the solution)

	while err > prec and cont< iterations :
		conv1=N0*fast_conv(log2(((1+f0)/(1+A0))),s_set,Lambda,N)
		f=-A*exp2(conv1)*exp2(logSstar(-gamma-kset)+logSstar(gamma-kset))
	#---------------------------	
		conv2=4*fast_conv(log2(((1- f)/(1-A))),s_set,Lambda,N)
		corr=a_1+conv1+conv2
		f0_tmp=-A0*(exp(corr))*exp2(logSstar(-gamma-kset)+logSstar(gamma-kset))
	#---------------------------

		gamma_tmp=exact_bethe(L,f0_tmp,f,n,gamma,A0,corr,Lambda,N,N0)
		gamma_plot.append(gamma_tmp)


	#get the error
		err=max(error(f0_tmp,f0),abs(gamma-gamma_tmp)*0.001)
		#print('Iteration:',cont)
		#print('Error',err)
		#print('Gamma:',gamma_tmp)


	#update
		f0=(1-damping_param)*f0_tmp+damping_param*f0
		gamma=gamma_tmp
		cont +=1
	#----------------------------- 


	if cont == iterations:
		print('Max iterations reached')
		if err > min_prec :
			print('Failed convergence')
			f0=zeros(N)
			f=zeros(N)
			gamma=0
			#quit()
	#----------------------------
	#-- Final output
	#print('The final gamma is:', gamma)
	#print('The energy in this state is', energy(L,f0,gamma,Lambda,N))
 
	#print('Iterations:',cont)
	#print('Energy:',energy(L,f0,gamma,Lambda,N))
	#print('Gamma:', gamma)
	#print('Asymptotic Energy:',4/cosh(asymptotic_bethe(L,n) ))
	#print('Asymptotic gamma:', asymptotic_bethe(L,n))
 
	return f0,f,gamma,array(gamma_plot)


def asy_tba_solver(L,n,Lambda=50,N=int(2**10)):
	#------------------------
	kset=arange(-Lambda,Lambda,(2*Lambda)/N)
	#------------------------

	#-------------------
	# 	INITIALIZATION 
	#-------------------

	gamma=asymptotic_bethe(L,n)    #initialize =>> asymptotic bethe roots
	#------------------
	#-- Driving term
	a_1 = -L*Emirr(kset)

	#------------------
	##############################
	##############################

	#-------------------
	#print('n=', n)
	#print('Asymptotic gamma:',gamma)


	f=exp2(logSstar(-gamma-kset)+logSstar(gamma-kset))
	#---------------------------	
	f0= exp2(a_1)*exp2(logSstar(-gamma-kset)+logSstar(gamma-kset))
	#---------------------------

	return f0,f,gamma
	
	
def energy(L,f0,gamma,Lambda,N,N0):
	kset=arange(-Lambda,Lambda,(2*Lambda)/N)
	dp=[2*cosh(j)/(sinh2(j,vec=False)**2) for j in kset] 
	dp[N//2]=0 #in the pole the integrand is zero
	integ=-(N0/(2*pi))*integrate.simps(dp*(log2(1+f0)),kset)	#no asymptotic problems due to exp suppression
	return (integ + 4/(cosh(gamma))).real
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
