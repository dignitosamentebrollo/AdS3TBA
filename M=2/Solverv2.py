import LoadSave_Libv2 as lslib
import TBAv2 as tba
#---------------------
from numpy import *

    
#--------------------------------------------------------

def full_solution(L,Lambda,N,N0,n1,n3,x0):
    sim_tag=f'L{L}_N{N}'
    foldername=f'N0{N0}'
    #RELOAD EVERY TIME THE ARRAY, AS EXPLAINED IN launcherv2.py
    with open('data/{}/{}/f0.npy'.format(foldername,sim_tag), 'rb') as w:
        f0=load(w)
    with open('data/{}/{}/f.npy'.format(foldername,sim_tag), 'rb') as w:
        f=load(w)
    with open('data/{}/{}/gamma1.npy'.format(foldername,sim_tag), 'rb') as w:
        gamma1=load(w)
    with open('data/{}/{}/gamma2.npy'.format(foldername,sim_tag), 'rb') as w:
        gamma2=load(w)
    with open('data/{}/{}/energy.npy'.format(foldername,sim_tag), 'rb') as w:
        en=load(w)
    with open('data/{}/{}/asy_gamma1.npy'.format(foldername,sim_tag), 'rb') as w:
        asy_gamma1=load(w)
    with open('data/{}/{}/asy_gamma2.npy'.format(foldername,sim_tag), 'rb') as w:
        asy_gamma2=load(w)
    with open('data/{}/{}/asy_en.npy'.format(foldername,sim_tag), 'rb') as w:
        asy_en=load(w)
        
    f0[n1,n3],f[n1,n3],gamma1[n1,n3],gamma2[n1,n3],asy_gamma1[n1,n3],asy_gamma2[n1,n3],asy_en[n1,n3]=tba.tba_solver(L,n1,n3,N0,Lambda=Lambda,N=N,x0=x0)
    en[n1,n3]=tba.energy(L,f0[n1,n3],gamma1[n1,n3],gamma2[n1,n3],Lambda,N,N0)
    
    #SAVE EVERY TIME THE RESULTS
    lslib.master_save(f0,f,gamma1,gamma2,en,folder_name='data/{}/{}'.format(foldername,sim_tag))
    lslib.single_save(asy_gamma1,tag='asy_gamma1',folder_name='data/{}/{}'.format(foldername,sim_tag))
    lslib.single_save(asy_gamma2,tag='asy_gamma2',folder_name='data/{}/{}'.format(foldername,sim_tag))
    lslib.single_save(asy_en,tag='asy_en',folder_name='data/{}/{}'.format(foldername,sim_tag))

    
    print('--Done--')
    
    return 0