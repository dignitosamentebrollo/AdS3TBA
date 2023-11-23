import os
import LoadSave_Libv2 as lslib
import TBAv2 as tba
#---------------------
from numpy import *

#======================
N=int(2**12) # rapidity sampling
Lambda=40	# rapidity cutoff
N0=2        # number of massless Y functions
#======================
L=16 #string length

sim_tag=f'L{L}_N{N}'
path=os.getcwd()
foldername=f'N0{N0}'
os.chdir('{}/data/{}'.format(path,foldername)) #create before the folder data/N0{N0}
os.makedirs('{}'.format(sim_tag),exist_ok=True)
os.chdir('../')

f0,f,gamma1,gamma2,en=lslib.Bethe_Set(N,L)
asy_f0,asy_f,asy_gamma1,asy_gamma2,asy_en=lslib.Bethe_Set(N,L)
kset=arange(-Lambda,Lambda,(2*Lambda)/N)


lslib.master_save(f0,f,gamma1,gamma2,en,folder_name='{}/{}'.format(foldername,sim_tag))
lslib.single_save(asy_f0,tag='asy_f0',folder_name='{}/{}'.format(foldername,sim_tag))
lslib.single_save(asy_f,tag='asy_f',folder_name='{}/{}'.format(foldername,sim_tag))
lslib.single_save(asy_gamma1,tag='asy_gamma1',folder_name='{}/{}'.format(foldername,sim_tag))
lslib.single_save(asy_gamma2,tag='asy_gamma2',folder_name='{}/{}'.format(foldername,sim_tag))
lslib.single_save(asy_en,tag='asy_en',folder_name='{}/{}'.format(foldername,sim_tag))
lslib.single_save(kset,tag='kset',folder_name='{}/{}'.format(foldername,sim_tag))