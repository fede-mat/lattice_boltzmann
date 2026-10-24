# Lecture on Lattice Boltzmann methods
# TU Munich, summer term 2026
# Prof. Dr. Barbara Wohlmuth, Dr. Stephan Lunowa
#
# Programming Tutorial 3b - LBM Implementation sugar ball dissolution

import os
import numpy as np
import time
from matplotlib import pyplot, colors
from numpy.random import normal
from pyevtk.hl import imageToVTK

# performance measurement
measure_performance = True
time_start = time.perf_counter()

# ---------------------------------------------------------------------------
# Geometry and Simulation Parameters
# ---------------------------------------------------------------------------
Nx = 400
Ny = 100
Nt = 10000
plot_interval = 25

# Obstacle Configuration
cyl_x    = Nx/5 + 1  # x-position of cylinder midpoint
cyl_y    = Ny/2 + 3  # y-position of cylinder midpoint (exact symmetry is avoided)
cyl_r    = Ny/10 + 1 # radius of cylinder
obstacle = ((np.arange(Nx) - cyl_x)**2)[:, None] + (np.arange(Ny) - cyl_y)**2 <= cyl_r**2 - 1
recession_progress = np.zeros((Nx, Ny))
recession_threshold = 10

### Exercise a) Physical parameters #########################################
Re      = 100                     # Reynolds number (main flow parameter)
uMax    = 0.1                     # Maximal velocity at channel inflow
Pr      = 0.5                     # Prandtl number (convection/diffusion ratio)
nu      = uMax * 2 * cyl_r / Re   # kinematic viscosity (according to Reynolds number)
D       = nu / Pr                 # sugar diffusivity (according to Prandtl number)
omega   = 1 / (3 * nu + 0.5)      # relaxation rate (velocity)
omega_C = 1 / (3 * D + 0.5)       # relaxation rate (sugar concentration)
print(f"nu = {nu}, D = {D}")
print(f"omega = {omega}, omega_C = {omega_C}")
#############################################################################

# ---------------------------------------------------------------------------
# Graphical sketch of the numbering in the LBM stencil
# ---------------------------------------------------------------------------
#
#  6   2    5
#    \ | /
#  3 - 0 - 1
#    / | \
#  7   4   8
#
c  = np.array([[0, 1, 0, -1,  0,  1, -1, -1,  1],
               [0, 0, 1,  0, -1,  1,  1, -1, -1]])
Nc = c.shape[1]
w  = np.ones((Nc,1,1)) / 9
w[0] = 4/9
w[5:] = 1/36
opp = np.array([[0, 3, 4, 1, 2, 7, 8, 5, 6]]).T

# Bounce Back directions on obstacle
def obstacle_bb_directions(obstacle):
    bb_cells, inner_cells = [[]], [[]]
    for i in range(1, Nc):
        bb_cells.append( np.roll(obstacle, c[:,i], axis=[0,1]) & ~obstacle )
        inner_cells.append( ~np.roll(obstacle, -c[:,i], axis=[0,1]) & obstacle )
    return bb_cells, inner_cells

bb_cells, inner_cells = obstacle_bb_directions(obstacle)

# Visualization
if not measure_performance:
	pyplot.figure(1, figsize=(12, 6), layout='tight')
	pyplot.draw()
	pyplot.pause(1e-6)

def visualize(u, C, obstacle):
	pyplot.clf()

	U = np.hypot(u[0], u[1])
	U[obstacle] = np.nan
	pyplot.subplot(211)
	pyplot.imshow(U.T, origin='lower', vmin=0, vmax=1.3*uMax)
	pyplot.title('Velocity magnitude |u|')
	pyplot.colorbar()

	pyplot.subplot(212)
	C[obstacle] = np.nan
	pyplot.imshow(C.T, origin='lower', cmap='jet', vmin=0, vmax=1)
	pyplot.title('Concentration C')
	pyplot.colorbar()

	pyplot.draw()
	pyplot.pause(1e-6)

def save_snapshot(u, C, obstacle, t):
	pyplot.figure(figsize=(12, 6), layout='tight')

	U = np.hypot(u[0], u[1])
	U[obstacle] = np.nan
	pyplot.subplot(211)
	pyplot.imshow(U.T, origin='lower', vmin=0, vmax=1.3*uMax)
	pyplot.title(f'Velocity magnitude |u| at t={t*25}')
	pyplot.colorbar()

	pyplot.subplot(212)
	C[obstacle] = np.nan
	pyplot.imshow(C.T, origin='lower', cmap='jet', vmin=0, vmax=1)
	pyplot.title(f'Concentration C at t={t*25}')
	pyplot.colorbar()

	os.makedirs("coding_session/second_problem/videos", exist_ok=True)
	pyplot.savefig(f"coding_session/second_problem/videos/snapshot_{t:05d}.png")
	pyplot.close()	

def equilibrium(rho, u):
	cu = 3 * np.tensordot(c.T, u, axes=(1,0))
	return w * rho * (1 + cu + 0.5*cu**2 - 1.5*np.sum(u**2, axis=0))

# Inflow profile (Poiseuille normed to max velocity of 1)
def vel_profile(y):
	return 4 * uMax / Ny**2 * (y + 0.5) * (Ny - 0.5 - y)
profile = vel_profile(np.arange(1,Ny-1))
uIn = np.array([[2/3, 1/6, 1/6]]).T * profile[None,:]

# Initialization of population arrays with slight perturbation of density
u = np.where(obstacle, np.zeros((2, Nx, Ny)),
             np.stack((vel_profile(np.arange(Ny)), np.zeros((Ny,))))[:,None,:])
f = equilibrium(1.0, u)
g = equilibrium(np.where(obstacle, 1.0, 0.0), u)

# Main time loop
time_loop_start = time.perf_counter()
for t in range(Nt):
	### Exercise b) Macroscopic quantities (flow) ###########################
	rho = np.sum(f, axis=0, dtype=float)
	moment = np.tensordot(c, f, axes=(1,0)) 
	u = moment / (rho + (rho == 0))[None, ...] 

	### Exercise c) Macroscopic concentration ###############################
	C = np.sum(g, axis=0, dtype=float)

	#########################################################################

	### Zou-He Scheme for In/Outflow Boundary Conditions ###
	# Inlet
	u[0, 0, 1:-1] = profile
	u[1, 0, 1:-1] = 0
	rho[0, 1:-1] = (np.sum(f[0:5:2, 0, 1:-1], axis=0) + 2 * np.sum(f[[3,6,7], 0, 1:-1], axis=0)) / (1 - profile)
	f[[1,5,8], 0, 1:-1] = f[[3,7,6], 0, 1:-1] + rho[0, 1:-1] * uIn
	f[[5,8], 0, 1:-1] += np.array([[-0.5], [0.5]]) * (f[2, 0, 1:-1] - f[4, 0, 1:-1])
	# Outlet
	rho[-1, 1:-1] = 1.0
	u[0, -1, 1:-1] = np.sum(f[0:5:2, -1, 1:-1], axis=0) + 2 * np.sum(f[[1,5,8], -1, 1:-1], axis=0) - 1
	u[1, -1, 1:-1] = 0
	f[[3,6,7], -1, 1:-1] = f[[1,8,5], -1, 1:-1] - np.array([[2/3, 1/6, 1/6]]).T * u[0, -1, 1:-1]
	f[[6,7], -1, 1:-1] += np.array([[-0.5], [0.5]]) * (f[2, -1, 1:-1] - f[4, -1, 1:-1])
	########################################################

	### Exercise b) Collision step (flow) ###################################
	fPost = f - omega * (f - equilibrium(rho, u))
	### Exercise c) Collision step (concentration) ##########################
	gPost = g - omega_C * (g - equilibrium(C, u))
	#########################################################################

	### Exercise b) Streaming step (flow) ###################################
	for i in range(Nc):
		f[i] = np.roll(fPost[i], c[:,i], axis=[0,1])
		### Exercise c) Streaming step (concentration) ######################
		g[i] = np.roll(gPost[i], c[:,i], axis=[0,1])
	#########################################################################

	### Exercise b) Wall Boundary Conditions (Bounce Back) ##################
	top = [2, 5, 6]
	for it, ib in zip(top, opp[top]):
		f[it, :, 0] = fPost[ib, :, 0]
		f[ib, :, -1] = fPost[it, :, -1]
		### Exercise d) Wall Boundary Conditions (Bounce Back) ##############
		g[it, :, 0] = gPost[ib, :, 0]
		g[ib, :, -1] = gPost[it, :, -1]
	#########################################################################

	### Exercise d) In/Outflow Boundary Conditions ##########################
	right = [1, 5, 8]
	for ir, il in zip(right, opp[right]):
		# Inlet
		g[ir, 0, :] = - gPost[il, 0, :] + 2 *  w[il]*0
		# Outlet
		g[il, -1, :] = - gPost[ir, -1, :]*0  
	#########################################################################

	### Exercise e) Obstacle Boundary Conditions (BB / Anti BB) #############
	for i in range(1, Nc):
		f[i, bb_cells[i]] = fPost[opp[i], bb_cells[i]]
		g[i, bb_cells[i]] = -gPost[opp[i], bb_cells[i]] + 2 * w[i] * 1
	#########################################################################

	### Exercise f) Outward Normal Approxmation #############################
	normal = 0.0
	for i in range(1, Nc):
		# Remark: the variable is automatically catsed the the correct type (float) due to the multiplication with c[:,i] (float) and bb_cells[i] (bool) 
		# it's always casted to the finest type of the operands, so the result is float. bool-> int->float. 
		normal += c[:,i][:,None,None] * bb_cells[i] 
	norm = np.hypot(normal[0], normal[1])
	normal /= norm + (norm < 1e-6)

	# # Persional Solution of Exercise f) Outward Normal Approxmation

	# # Compuute the boundary cells (maybe this row is reduntant, but it is here to make sure that the boundary cells are computed correctly)
	# bb_cells, inner_cells = obstacle_bb_directions(obstacle)
	# boundary_cells = np.zeros_like(obstacle, dtype=bool)
	# for bb_cell in bb_cells:
	# 	boundary_cells |= bb_cell.astype(bool)

	# # Compute the gradient of the obstacle array
	# gradient_x, gradient_y = np.gradient(obstacle.astype(float))

	# # Normalize the gradient to obtain the normal vectors
	# norm = np.sqrt(gradient_x**2 + gradient_y**2)
	# normal_x = np.zeros_like(gradient_x)
	# normal_y = np.zeros_like(gradient_y)

	# # Assign the normal vectors only to the boundary cells
	# normal_x[boundary_cells] = gradient_x[boundary_cells] / norm[boundary_cells]
	# normal_y[boundary_cells] = gradient_y[boundary_cells] / norm[boundary_cells]

	# # Merge the normal vectors into a single array
	# normal = np.stack((normal_x, normal_y), axis=0)

	#########################################################################

	### Exercise g) Normal Gradient Approxmation ############################
	D_grad_C = D * omega_C * 3 * np.tensordot(c, g, axes=(1,0))  # - D ∇ C

	v_n = - np.sum(normal * D_grad_C, axis=0)

	# print(f"v_n size: {v_n.shape}, normal size: {normal.shape}, D_grad_C size: {D_grad_C.shape}	")

	## Then it is shifted from the bb_cells to the inner cells
	w_inner = np.zeros((Nx, Ny))
	for i in range(1, Nc):
		w_inner += inner_cells[i]
		v_n[inner_cells[i]] += v_n[bb_cells[i]]
	mask = w_inner > 0
	v_n[mask] /= w_inner[mask]
	#########################################################################

	### Exercise h) Recession Dynamics ######################################
	recession_progress[mask] -= v_n[mask]
	breakaway = obstacle & (recession_progress > recession_threshold)
	if np.any(breakaway):
		obstacle[breakaway] = False
		bb_cells, inner_cells = obstacle_bb_directions(obstacle)
	#########################################################################

	# Graphical output
	if not measure_performance and (t % plot_interval == 0 or t+1 == Nt):
		visualize(u, C, obstacle)
		
	if measure_performance and (t % plot_interval == 0 or t+1 == Nt):
		save_snapshot(u, C, obstacle, t//plot_interval)
		# ffmpeg -framerate 30 -i snapshot%05d.png -c:v libx264 -pix_fmt yuv420p output_{i}.mp4

time_loop_end = time.perf_counter()

# VTK output
os.makedirs("coding_session/second_problem/VTKOutput", exist_ok=True)
ux = u[:1].copy()
uy = u[1:].copy()
imageToVTK("coding_session/second_problem/VTKOutput/SBD_data", (0,0,0), (1,1,1),
           cellData = {"density": rho[..., None], "u_x": ux, "u_y": uy, "concentration": C[..., None]})

# Performance output
time_end = time.perf_counter()
lattice_updates = 2 * Nx * Ny * Nt
print(f"Total time: {time_end-time_start} s")
print(f"Performance: {lattice_updates / 1.0e6 / (time_loop_end-time_loop_start):.3f} MLUPS")
