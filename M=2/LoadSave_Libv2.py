from numpy import *
#============================
def Bethe_Set(N,L):
	f0=array([array([zeros(N)]*(L//2+1))]*(L//2+1),dtype=complex)
	f=array([array([zeros(N)]*(L//2+1))]*(L//2+1),dtype=complex)
	gamma1=array([zeros(L//2+1)]*(L//2+1),dtype=complex)
	gamma2=array([zeros(L//2+1)]*(L//2+1),dtype=complex)
	en=array([zeros(L//2+1)]*(L//2+1),dtype=complex)

	return  f0,f,gamma1,gamma2,en
#==============================================================
# Note: NEED TO CREATE BEFOREHAND A FOLDER CALLED 'data' in the current Directory
#==============================================================
def single_save(array, tag='test',folder_name='data'):
	with open('{}/{}.npy'.format(folder_name,tag), 'wb') as f:
		save(f,array)

def single_filesave(array, tag='test',folder_name='data'):
	with open('{}/{}.dat'.format(folder_name,tag), 'wb') as f:
		savetxt(f,array)

def single_load(tag='test',folder_name='data'):
	with open('{}/{}.npy'.format(folder_name,tag), 'rb') as f:
		array=load(f)
	return array
#===============================================================
def master_save(f0,f,gamma1,gamma2,en,folder_name='data'):
	single_save(f0,tag='f0',folder_name=folder_name)
	single_save(f,tag='f',folder_name=folder_name)
	single_save(gamma1,tag='gamma1',folder_name=folder_name)
	single_save(gamma2,tag='gamma2',folder_name=folder_name)
	single_save(en,tag='energy',folder_name=folder_name)

#-------------------------------
def master_load(folder_name='data'):
	n_S1=single_load(tag='n_S1',folder_name=folder_name)
	n_S2=single_load(tag='n_S2',folder_name=folder_name)
	v_S1=single_load(tag='v_S1',folder_name=folder_name)
	v_S2=single_load(tag='v_S2',folder_name=folder_name)
	rho_S1=single_load(tag='rho_S1',folder_name=folder_name)
	rho_S2=single_load(tag='rho_S2',folder_name=folder_name)
	u=single_load(tag='u',folder_name=folder_name)

	return n_S1,n_S2,v_S1,v_S2,rho_S1,rho_S2,u
#-----------------
