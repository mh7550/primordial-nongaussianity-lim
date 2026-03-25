import numpy as np
import time
from colossus.cosmology import cosmology as cosmo_colossus
cosmo_colossus.setCosmology('planck18')
from colossus.lss import bias
from colossus.lss import mass_function
from pylab import figure,savefig
import matplotlib.pyplot as plt
params=cosmo_colossus.getCurrent()
print(params.H0)
cosmo = cosmo_colossus.getCurrent()
zi=np.arange(131)*0.1
chii=cosmo.comovingDistance(z_max=zi)
lam0=np.array([0.3727,0.4861,0.5007,0.6563])
lamband=np.array([0.75,1.10,1.63,2.42,3.82,4.42,5.00])
dlamband=np.diff(lamband)
lamcen=0.5*(lamband[:-1]+lamband[1:])
R=np.array([41,41,41,35,110,130])
dlamchan=lamcen/R
nchan=np.floor(dlamband/dlamchan)
nchan=nchan.astype(int)
nchantot=np.sum(nchan)
lamchan=np.array([lamband[0]])
for i in np.arange(6):
    lamchan=np.concatenate((lamchan,np.linspace(lamband[i]+dlamband[i]/nchan[i],lamband[i+1],nchan[i])))
lamchan1=lamchan[:-1]
lamchan2=lamchan[1:]
lamchanm=0.5*(lamchan1+lamchan2)
dlamchan=lamchan2-lamchan1
print(nchantot)
zlam=np.outer(1./lam0,lamchanm)-1.
chilam=np.zeros((4,nchantot))
for i in np.arange(4):
    chilam[i]=np.interp(zlam[i],zi,chii,left=0.)
Ez=cosmo.Ez(zlam)
limber=np.round((chilam/3000.)*Ez*np.outer(lam0,1./dlamchan))
limber=np.max(limber,axis=0)
np.savetxt('test_limber.txt',limber,fmt='%d')
