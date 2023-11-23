from numpy import *
#============================
def Bethe_Set(N,L):
	f0=array([zeros(N)]*(L//2+1),dtype=complex)
	f=array([zeros(N)]*(L//2+1),dtype=complex)
	gamma=zeros(L//2+1,dtype=complex)
	en=zeros(L//2+1,dtype=complex)

	return  f0,f,gamma,en
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
def master_save(f0,f,gamma,en,folder_name='data'):
	single_save(f0,tag='f0',folder_name=folder_name)
	single_save(f,tag='f',folder_name=folder_name)
	single_save(gamma,tag='gamma',folder_name=folder_name)
	single_save(en,tag='energy',folder_name=folder_name)