import numpy as np
import matplotlib.pyplot as plt

"""
Modified by Oscar Amaro from 2D to 1D following
https://levelup.gitconnected.com/create-your-own-quantum-mechanics-simulation-with-python-51e215346798
https://github.com/pmocz/quantumspectral-python

Create Your Own Quantum Mechanics Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate the Schrodinger-Poisson system with the Spectral method
"""

def simulaten(n):
	""" Quantum simulation """
	
	# Simulation parameters
	N         = int(2**n)    # Spatial resolution
	t         = 0      # current time of the simulation
	tEnd      = 3.00    # time at which simulation ends
	dt        = 0.01  # timestep
	tOut      = 10  # draw frequency
	tdim = int(np.ceil(tEnd/dt/tOut))
	G         = 1/(4*np.pi)  # Gravitaitonal constant

	# Domain [0,8]
	rhot = np.zeros((N,tdim))
	print(np.shape(rhot))
	L = 8
	xx = np.linspace(0,L, num=N+1)  # Note: x=0 & x=1 are the same point!
	xx = xx[0:N]                     # chop off periodic point
	
	# Intial Condition
	amp = 0.01
	sigma = 0.03
	rho = 1.0+0.6*np.sin(np.pi*xx/4)
	# normalize wavefunction to <|psi|^2>=1
	rhobar = np.mean( rho )
	rho /= rhobar
	psi = np.sqrt(rho)
	
	# Fourier Space Variables
	klin = 2.0 * np.pi / L * np.arange(-N/2, N/2)
	kx = np.fft.ifftshift(klin)
	kSq = kx**2
	
	# Potential
	Vhat = -np.fft.fftn(4.0*np.pi*G*(np.abs(psi)**2-1.0)) / ( kSq  + (kSq==0))
	V = np.real(np.fft.ifftn(Vhat))
	
	# number of timesteps
	Nt = int(np.ceil(tEnd/dt))
	
	# prep figure
	fig = plt.figure(figsize=(6,4), dpi=80)
	outputCount = 0
	
	# Simulation Main Loop
	for i in range(Nt):
		# (1/2) kick
		psi = np.exp(-1.j*dt/2.0*V) * psi
		
		# drift
		psihat = np.fft.fftn(psi)
		psihat = np.exp(dt * (-1.j*kSq/2.))  * psihat
		psi = np.fft.ifftn(psihat)
		
		# update potential
		Vhat = -np.fft.fftn(4.0*np.pi*G*(np.abs(psi)**2-1.0)) / ( kSq  + (kSq==0))
		V = np.real(np.fft.ifftn(Vhat))
		
		# (1/2) kick
		psi = np.exp(-1.j*dt/2.0*V) * psi
		
		# update time
		t += dt
		
		# save
		if i%tOut==0:
			rhot[:,outputCount] = np.abs(psi)**2
			outputCount += 1
			
	cs = plt.imshow(np.rot90(rhot),  extent=[0,8,0,3], aspect=L/tEnd , cmap = 'hot')
	clb = fig.colorbar(cs, shrink=0.9)
	clb.set_label(r'$|\psi|^2$', labelpad=0, y=1.05, rotation=0)
	plt.xlabel('x')
	plt.xlabel('t')
	plt.title(r'n = {}'.format(n))
	plt.savefig('n{}'.format(n),dpi=240)
	#plt.show()
	

def main():
	for n in range(2,9):
		simulaten(n)
	return 0
	

if __name__== "__main__":
  main()
