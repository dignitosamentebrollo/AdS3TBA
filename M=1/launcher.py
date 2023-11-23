import os
import Solver as Solver
import LoadSave_Lib as lslib
import time
#---------------------
from numpy import *
#======================
#======================
N=int(2**12) # rapidity sampling
Lambda=40	# rapidity cutoff
N0=2        # number of massless Y functions
#======================
Lset=array([8]) #values of string length to compute

for L in Lset:
    print('String length L=',L)
    print('Number of Y0 functions N0=',N0)
    
    #==================
    #== SAVING DIRECTORY
    sim_tag=f'L{L}_N{N}'
    path=os.getcwd()
    foldername=f'N0{N0}'
    os.chdir('{}/data/{}'.format(path,foldername))
    os.makedirs('{}'.format(sim_tag),exist_ok=True)
    os.chdir('../')
    #==================

    #==============================
    kset=arange(-Lambda,Lambda,(2*Lambda)/N) #rapidity set from -Lambda to Lambda

    #================
    #== SOLVING
    #================
    t0=time.time()

    # Solution of the complete model
    f0,f,gamma,en=Solver.full_solution(L,Lambda,N,N0)
    # Solution of the asymptotic model (no convolutions)
    asy_f0,asy_f,asy_gamma,asy_en=Solver.asymptotic_solution(L,Lambda,N)


    #================
    #== SAVING
    #================
    lslib.master_save(f0,f,gamma,en,folder_name='{}/{}'.format(foldername,sim_tag))
    #lslib.single_save(list_gamma_plot,tag='list_gamma_plot',folder_name='data/{}'.format(sim_tag))
    lslib.single_save(asy_f0,tag='asy_f0',folder_name='{}/{}'.format(foldername,sim_tag))
    lslib.single_save(asy_f,tag='asy_f',folder_name='{}/{}'.format(foldername,sim_tag))
    lslib.single_save(asy_gamma,tag='asy_gamma',folder_name='{}/{}'.format(foldername,sim_tag))
    lslib.single_save(asy_en,tag='asy_en',folder_name='{}/{}'.format(foldername,sim_tag))
    lslib.single_save(kset,tag='kset',folder_name='{}/{}'.format(foldername,sim_tag))
    lslib.single_filesave(en,tag='en',folder_name='{}/{}'.format(foldername,sim_tag))
    lslib.single_filesave(asy_en,tag='asy_en',folder_name='{}/{}'.format(foldername,sim_tag))
    lslib.single_filesave(gamma,tag='gamma',folder_name='{}/{}'.format(foldername,sim_tag))
    lslib.single_filesave(asy_gamma,tag='asy_gamma',folder_name='{}/{}'.format(foldername,sim_tag))

    # ---------------------------
    print('--All data are saved--')
    print('Elapsed time:',time.time()-t0)

    #================
    #== PLOTTING
    #================
    nset=arange(1,L//2+1,1)
    import matplotlib.pyplot as plt
    plt.axes()
    plt.plot(linspace(0,L//2,2**10),4*sin((pi*linspace(0,L//2,2**10))/L),'c',linestyle='dashed',label=r'Free')
    plt.plot(nset,en[1::].real,'k.',label=r'Exact')
    plt.plot(nset,asy_en[1::].real,'rx',label=r'Asymptotic')

    # x axis
    plt.xlabel('Mode',fontsize=12)
    if L<100:
        a=2
    elif L>=500:
        a=10
    else: a=5
    plt.xticks(range(0,L//2+1,a))

    #y axis
    plt.ylabel('Energy',fontsize=11)
    plt.title(f'L={L}')

    plt.legend()
    plt.tight_layout()
    os.chdir('{}/data/{}/{}'.format(path,foldername,sim_tag))
    plt.savefig(f'EnergyL{L}.pdf')
    plt.close()
    os.chdir('../')
    os.chdir('../')

# --------------------------
print('--end--')