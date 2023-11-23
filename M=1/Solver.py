import LoadSave_Lib as lslib
import TBA as tba
#---------------------
from numpy import *

def full_solution(L,Lambda,N,N0):
	f0,f,gamma,en=lslib.Bethe_Set(N,L) #creates the arrays
	#----------------------------------------

	[A,A0]=tba.asymptotics(N0) #asymptotics
	print('Asymptotics for Y and Y0:', [A,A0])
	#list_gamma_plot=[]
	#---------------------------------

	#------Solution of the tba--------
	for n in range(1,L//2+1): #iteration over the modes of the excitations
		f0[n],f[n],gamma[n],gamma_plot=tba.tba_solver(L,n,A,A0,N0,Lambda=Lambda,N=N)
		#list_gamma_plot.append(gamma_plot)
		#------compute the energy
		en[n]=tba.energy(L,f0[n],gamma[n],Lambda,N,N0)

	print('--Done--')


	return f0,f,gamma,en


def asymptotic_solution(L,Lambda,N):
	asy_f0,asy_f,asy_gamma,asy_en=lslib.Bethe_Set(N,L)
	#----------------------------------------

	#------Solution of the tba--------
	for n in range(1,L//2+1):
		asy_f0[n],asy_f[n],asy_gamma[n]=tba.asy_tba_solver(L,n,Lambda=Lambda,N=N)
		#------compute the energy
		asy_en[n]=4/cosh(asy_gamma[n])

	print('--Asymptotic done--')
		
	return asy_f0,asy_f,asy_gamma,asy_en