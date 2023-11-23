import Solverv2 as Solver
import LoadSave_Libv2 as lslib
import time
#---------------------
from numpy import *

#======================
N=int(2**12) # rapidity sampling
Lambda=40	# rapidity cutoff
N0=2        # number of massless Y functions
#======================
L=16 #worldsheet lenght

#==DIRECTORY
sim_tag=f'L{L}_N{N}'
foldername=f'N0{N0}'
#==================

print('L=',L)
print('N0=',N0)


# BEFORE RUNNING THIS CODE, RUN makedir.py WHICH CREATES FOLDERS AND ARRAYS

# THIS BECAUSE COMPARED TO THE M=1 CASE, THE CODE IS VERY SENSITIVE TO THE
# STARTING POINT FOR SOLVING THE BETHE EQUATIONS, THUS SOMETIMES FAILS

# FOR THIS REASON, I CREATE BEFORE THE ARRAYS AND EACH TIME I LOAD THEM
# AND SAVE EACH RESULT I GET

#================
#== SOLVING
#================
t0=time.time()

# AS STARTING POINT FOR SOLVING BETHE EQUATIONS FOR (n1,n3)
# I TAKE TAKE THE ASYMPTOTIC VALUES FOR (n1,n3-1)
# FOR (n1,n3)=(1,1) I ARBITRARY CHOSE x0 BELOW
x0=[3.2,3] #for n1=n3=1

#LOAD THE ALREADY COMPUTED ASYMPTOTIC BETHE ROOTS
with open('data/{}/{}/asy_gamma1.npy'.format(foldername,sim_tag), 'rb') as w:
    asy_gamma1=load(w)
with open('data/{}/{}/asy_gamma2.npy'.format(foldername,sim_tag), 'rb') as w:
    asy_gamma2=load(w)


ex=arange(1,L//2+1) #array with all possible modes

for n1 in [1,2,3,4]: #modes for which I decide to compute the energies
    if n1!=1:
        x0=[asy_gamma2[1,n1].real,asy_gamma1[1,n1].real]
    for n3 in ex:
            # SINCE IT CAN FAILS FOR SOME (n1,n3) I SAVE AND RELOAD EVERY TIME
            with open('data/{}/{}/asy_gamma1.npy'.format(foldername,sim_tag), 'rb') as w:
                asy_gamma1=load(w)
            with open('data/{}/{}/asy_gamma2.npy'.format(foldername,sim_tag), 'rb') as w:
                asy_gamma2=load(w)
            if (asy_gamma1[n1,n3-1]!=0 and asy_gamma2[n1,n3-1]!=0): #setting the starting point for the solution of the bethe equations
                rev=False
                if n3>n1: rev=True
                x0=sorted([asy_gamma1[n1,n3-1].real,asy_gamma2[n1,n3-1].real],reverse=rev)
            Solver.full_solution(L,Lambda,N,N0,n1,n3,x0)



# ---------------------------
print('--All data are saved--')
print('Elapsed time:',time.time()-t0)



# --------------------------
print('--end--')