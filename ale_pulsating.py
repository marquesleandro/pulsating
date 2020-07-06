# =======================
# Importing the libraries
# =======================

import os
initial_path = os.getcwd()

import sys
directory = './lib_class'
sys.path.insert(0, directory)

from tqdm import tqdm
from time import time

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg

import search_file
import import_msh
import assembly
import ale
import benchmark_problems
import semi_lagrangian
import export_vtk
import relatory



print '''
               COPYRIGHT                    
 ======================================
 Simulator: %s
 created by Leandro Marques at 02/2019
 e-mail: marquesleandro67@gmail.com
 Gesar Search Group
 State University of the Rio de Janeiro
 ======================================
''' %sys.argv[0]



print ' ------'
print ' INPUT:'
print ' ------'
print ""


# ----------------------------------------------------------------------------
print ' (1) - Linear Element'
print ' (2) - Mini Element'
print ' (3) - Quadratic Element'
print ' (4) - Cubic Element'
polynomial_option = int(raw_input(" Enter polynomial degree option above: "))
print ""
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
print ' 3 Gauss Points'
print ' 4 Gauss Points'
print ' 6 Gauss Points'
print ' 12 Gauss Points'
gausspoints = int(raw_input(" Enter Gauss Points Number option above: "))
print ""
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
print ' (1) - Taylor Galerkin Scheme'
print ' (2) - Semi Lagrangian Scheme'
scheme_option = int(raw_input(" Enter simulation scheme option above: "))
print ""
print ""
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
nt = int(raw_input(" Enter number of time interations (nt): "))
directory_save = raw_input(" Enter folder name to save simulations: ")
print ""
# ----------------------------------------------------------------------------




print ' ------------'
print ' IMPORT MESH:'
print ' ------------'

start_time = time()

# Linear Element
if polynomial_option == 1:
 mesh_name = 'malha_poiseuille_ALE.msh'
 equation_number = 4

 directory = search_file.Find(mesh_name)
 if directory == 'File not found':
  sys.exit()

 msh = import_msh.Linear2D(directory, mesh_name, equation_number)
 msh.coord()
 msh.ien()


# Mini Element
elif polynomial_option == 2:
 mesh_name = 'malha_half_poiseuille.msh'
 equation_number = 4

 directory = search_file.Find(mesh_name)
 if directory == 'File not found':
  sys.exit()

 msh = import_msh.Mini2D(directory, mesh_name, equation_number)
 msh.coord()
 msh.ien()

# Quad Element
elif polynomial_option == 3:
 mesh_name = 'malha_half_poiseuille_quad.msh'
 equation_number = 4
 
 directory = search_file.Find(mesh_name)
 if directory == 'File not found':
  sys.exit()

 msh = import_msh.Quad2D(directory, mesh_name, equation_number)
 msh.coord()
 msh.ien()

# Cubic Element
elif polynomial_option == 4:
 mesh_name = 'malha_half_poiseuille_cubic.msh'
 equation_number = 4

 directory = search_file.Find(mesh_name)
 if directory == 'File not found':
  sys.exit()

 msh = import_msh.Cubic2D(directory, mesh_name, equation_number)
 msh.coord()
 msh.ien()



npoints                = msh.npoints
nelem                  = msh.nelem
x                      = msh.x
y                      = msh.y
IEN                    = msh.IEN
neumann_edges          = msh.neumann_edges
dirichlet_pts          = msh.dirichlet_pts
neighbors_nodes        = msh.neighbors_nodes
neighbors_elements     = msh.neighbors_elements
far_neighbors_nodes    = msh.far_neighbors_nodes
far_neighbors_elements = msh.far_neighbors_elements
length_min             = msh.length_min
GL                     = msh.GL
nphysical              = msh.nphysical 


CFL = 0.5
#dt = float(CFL*length_min)
dt = 0.03
Re = 100.0
Sc = 1.0

end_time = time()
import_mesh_time = end_time - start_time
print ' time duration: %.1f seconds' %import_mesh_time
print ""




print ' ---------'
print ' ASSEMBLY:'
print ' ---------'

start_time = time()


Kxx, Kxy, Kyx, Kyy, K, M, MLump, Gx, Gy, polynomial_order = assembly.Element2D(polynomial_option, GL, npoints, nelem, IEN, x, y, gausspoints)


end_time = time()
assembly_time = end_time - start_time
print ' time duration: %.1f seconds' %assembly_time
print ""





print ' --------------------------------'
print ' INITIAL AND BOUNDARY CONDITIONS:'
print ' --------------------------------'

start_time = time()

# Linear Element
if polynomial_option == 1:
 # ------------------------ Boundaries Conditions ----------------------------------
 # Applying vx condition
 xvelocity_LHS0 = sps.lil_matrix.copy(M)
 condition_xvelocity = benchmark_problems.Poiseuille(nphysical,npoints,x,y)
 condition_xvelocity.neumann_condition(neumann_edges[1])
 condition_xvelocity.dirichlet_condition(dirichlet_pts[1])
 condition_xvelocity.gaussian_elimination(xvelocity_LHS0,neighbors_nodes)
 vorticity_ibc = condition_xvelocity.ibc
 benchmark_problem = condition_xvelocity.benchmark_problem

 # Applying vy condition
 yvelocity_LHS0 = sps.lil_matrix.copy(M)
 condition_yvelocity = benchmark_problems.Poiseuille(nphysical,npoints,x,y)
 condition_yvelocity.neumann_condition(neumann_edges[2])
 condition_yvelocity.dirichlet_condition(dirichlet_pts[2])
 condition_yvelocity.gaussian_elimination(yvelocity_LHS0,neighbors_nodes)

 # Applying psi condition
 streamfunction_LHS0 = sps.lil_matrix.copy(K)
 condition_streamfunction = benchmark_problems.Poiseuille(nphysical,npoints,x,y)
 condition_streamfunction.streamfunction_condition(dirichlet_pts[3],streamfunction_LHS0,neighbors_nodes)
 # ---------------------------------------------------------------------------------

# Mini Element
elif polynomial_option == 2:
 # ------------------------ Boundaries Conditions ----------------------------------
 # Applying vx condition
 xvelocity_LHS0 = sps.lil_matrix.copy(M)
 condition_xvelocity = benchmark_problems.Poiseuille(nphysical,npoints,x,y)
 condition_xvelocity.neumann_condition(neumann_edges[1])
 condition_xvelocity.dirichlet_condition(dirichlet_pts[1])
 condition_xvelocity.gaussian_elimination(xvelocity_LHS0,neighbors_nodes)
 vorticity_ibc = condition_xvelocity.ibc
 benchmark_problem = condition_xvelocity.benchmark_problem

 # Applying vy condition
 yvelocity_LHS0 = sps.lil_matrix.copy(M)
 condition_yvelocity = benchmark_problems.Poiseuille(nphysical,npoints,x,y)
 condition_yvelocity.neumann_condition(neumann_edges[2])
 condition_yvelocity.dirichlet_condition(dirichlet_pts[2])
 condition_yvelocity.gaussian_elimination(yvelocity_LHS0,neighbors_nodes)

 # Applying psi condition
 streamfunction_LHS0 = sps.lil_matrix.copy(K)
 condition_streamfunction = benchmark_problems.Poiseuille(nphysical,npoints,x,y)
 condition_streamfunction.streamfunction_condition(dirichlet_pts[3],streamfunction_LHS0,neighbors_nodes)
 # ---------------------------------------------------------------------------------


# Quad Element
elif polynomial_option == 3:
 # ------------------------ Boundaries Conditions ----------------------------------
 # Applying vx condition
 xvelocity_LHS0 = sps.lil_matrix.copy(M)
 condition_xvelocity = benchmark_problems.QuadPoiseuille(nphysical,npoints,x,y)
 condition_xvelocity.neumann_condition(neumann_edges[1])
 condition_xvelocity.dirichlet_condition(dirichlet_pts[1])
 condition_xvelocity.gaussian_elimination(xvelocity_LHS0,neighbors_nodes)
 vorticity_ibc = condition_xvelocity.ibc
 benchmark_problem = condition_xvelocity.benchmark_problem

 # Applying vy condition
 yvelocity_LHS0 = sps.lil_matrix.copy(M)
 condition_yvelocity = benchmark_problems.QuadPoiseuille(nphysical,npoints,x,y)
 condition_yvelocity.neumann_condition(neumann_edges[2])
 condition_yvelocity.dirichlet_condition(dirichlet_pts[2])
 condition_yvelocity.gaussian_elimination(yvelocity_LHS0,neighbors_nodes)

 # Applying psi condition
 streamfunction_LHS0 = sps.lil_matrix.copy(K)
 condition_streamfunction = benchmark_problems.QuadPoiseuille(nphysical,npoints,x,y)
 condition_streamfunction.streamfunction_condition(dirichlet_pts[3],streamfunction_LHS0,neighbors_nodes)
 # ---------------------------------------------------------------------------------


# -------------------------- Initial condition ------------------------------------
vx = np.copy(condition_xvelocity.bc_1)
vy = np.copy(condition_yvelocity.bc_1)
psi = np.copy(condition_streamfunction.bc_1)
w = np.zeros([npoints,1], dtype = float)




#---------- Step 1 - Compute the vorticity and stream field --------------------
# -----Vorticity initial-----
vorticity_RHS = sps.lil_matrix.dot(Gx,vy) - sps.lil_matrix.dot(Gy,vx)
vorticity_LHS = sps.lil_matrix.copy(M)
w = scipy.sparse.linalg.cg(vorticity_LHS,vorticity_RHS,w, maxiter=1.0e+05, tol=1.0e-05)
w = w[0].reshape((len(w[0]),1))


# -----Streamline initial-----
# psi condition
streamfunction_RHS = sps.lil_matrix.dot(M,w)
streamfunction_RHS = np.multiply(streamfunction_RHS,condition_streamfunction.bc_2)
streamfunction_RHS = streamfunction_RHS + condition_streamfunction.bc_dirichlet
psi = scipy.sparse.linalg.cg(condition_streamfunction.LHS,streamfunction_RHS,psi, maxiter=1.0e+05, tol=1.0e-05)
psi = psi[0].reshape((len(psi[0]),1))
#----------------------------------------------------------------------------------


end_time = time()
bc_apply_time = end_time - start_time
print ' time duration: %.1f seconds' %bc_apply_time
print ""





print ' -----------------------------'
print ' PARAMETERS OF THE SIMULATION:'
print ' -----------------------------'

print ' Mesh: %s' %mesh_name
print ' Number of equation: %s' %equation_number
print ' Number of nodes: %s' %npoints
print ' Number of elements: %s' %nelem
print ' Smallest edge length: %f' %length_min
print ' Time step: %s' %dt
print ' Number of time iteration: %s' %nt
print ' Reynolds number: %s' %Re
print ' Schmidt number: %s' %Sc
print ""






print ' ----------------------------'
print ' SOLVE THE LINEARS EQUATIONS:'
print ' ----------------------------'
print ""
print ' Saving simulation in %s' %directory_save
print ""



start_time = time()
os.chdir(initial_path)



vorticity_bc_1 = np.zeros([npoints,1], dtype = float) 
for t in tqdm(range(0, nt)):
#for t in range(0, nt):
     
 # Linear and Mini Elements
 if polynomial_option == 1 or polynomial_option == 2:   
  # ------------------------ Export VTK File ---------------------------------------
  save = export_vtk.Linear2D(x,y,IEN,npoints,nelem,w,w,psi,vx,vy)
  save.create_dir(directory_save)
  save.saveVTK(directory_save + str(t))
  # --------------------------------------------------------------------------------

 # Quad Element
 elif polynomial_option == 3:   
  # ------------------------ Export VTK File ---------------------------------------
  save = export_vtk.Quad2D(x,y,IEN,npoints,nelem,w,w,psi,vx,vy)
  save.create_dir(directory_save)
  save.saveVTK(directory_save + str(t))
  # --------------------------------------------------------------------------------


 # ------------------------- Pulsating Boundary ------------------------------------
 y_boundary = np.zeros([npoints,1], dtype = float)
 for i in range(0,len(dirichlet_pts[1])):
  line = dirichlet_pts[1][i][0] - 1
  v1 = dirichlet_pts[1][i][1] - 1
  v2 = dirichlet_pts[1][i][2] - 1

  # oscillatory parameters ok
  # center point x=5.3
  if line == 0:
   y_boundary[v1] = 2.0*0.01*np.sin((2.0*np.pi/7.0)*x[v1])*np.cos((2.0*np.pi/32.0)*t)
   y_boundary[v2] = 2.0*0.01*np.sin((2.0*np.pi/7.0)*x[v2])*np.cos((2.0*np.pi/32.0)*t)

  elif line == 3:
   y_boundary[v1] = -2.0*0.01*np.sin((2.0*np.pi/7.0)*x[v1])*np.cos((2.0*np.pi/32.0)*t)
   y_boundary[v2] = -2.0*0.01*np.sin((2.0*np.pi/7.0)*x[v2])*np.cos((2.0*np.pi/32.0)*t)





 y = y + y_boundary
 # --------------------------------------------------------------------------------



 # ------------------------- ALE Scheme --------------------------------------------
 # Linear Element
 if polynomial_option == 1:
  k_lagrangian = 0.0
  k_laplace = 1.0
  
  vx_smooth, vy_smooth = ale.Laplacian_smoothing(neighbors_nodes, npoints, x, y, dt)

  vx_Ale = k_lagrangian*vx + k_laplace*vx_smooth
  vy_Ale = k_lagrangian*vy + k_laplace*vy_smooth


  for i in range(0,len(dirichlet_pts[4])):
   v1 = dirichlet_pts[4][i][1] - 1
   v2 = dirichlet_pts[4][i][2] - 1

   vx_Ale[v1] = 0.0
   vy_Ale[v1] = 0.0

   vx_Ale[v2] = 0.0
   vy_Ale[v2] = 0.0

  x = x + vx_Ale*dt
  y = y + vy_Ale*dt

  vx_SL = vx - vx_Ale
  vy_SL = vy - vy_Ale



 # Quad Element
 elif polynomial_option == 3:
  vx_Ale, vy_Ale = ale.Quadrotate(npoints, nelem, IEN, t, dirichlet_pts[4])

  vx_SL = vx - vx_Ale
  vy_SL = vy - vy_Ale

  x = x + vx_Ale*dt
  y = y + vy_Ale*dt
 # --------------------------------------------------------------------------------



 # ------------------------- Assembly --------------------------------------------
 print "\n"
 print "Assembly: "
 Kxx, Kxy, Kyx, Kyy, K, M, MLump, Gx, Gy, polynomial_order = assembly.Element2D(polynomial_option, GL, npoints, nelem, IEN, x, y, gausspoints)
 print "\n"
 print ' ----------------------------'
 print ' SOLVE THE LINEARS EQUATIONS:'
 print ' ----------------------------'
 # --------------------------------------------------------------------------------




 # ------------------------ Boundaries Conditions ----------------------------------
 # ---------------------------------------------------------------------------------
 # Linear Element
 if polynomial_option == 1:
  # Applying vx condition
  xvelocity_LHS0 = sps.lil_matrix.copy(M)
  condition_xvelocity = benchmark_problems.Poiseuille(nphysical,npoints,x,y)
  condition_xvelocity.neumann_condition(neumann_edges[1])
  condition_xvelocity.dirichlet_condition(dirichlet_pts[1])
  condition_xvelocity.gaussian_elimination(xvelocity_LHS0,neighbors_nodes)
  vorticity_ibc = condition_xvelocity.ibc
  benchmark_problem = condition_xvelocity.benchmark_problem
 
  # Applying vy condition
  yvelocity_LHS0 = sps.lil_matrix.copy(M)
  condition_yvelocity = benchmark_problems.Poiseuille(nphysical,npoints,x,y)
  condition_yvelocity.neumann_condition(neumann_edges[2])
  condition_yvelocity.dirichlet_condition(dirichlet_pts[2])
  condition_yvelocity.gaussian_elimination(yvelocity_LHS0,neighbors_nodes)
 
  # Applying psi condition
  streamfunction_LHS0 = sps.lil_matrix.copy(K)
  condition_streamfunction = benchmark_problems.Poiseuille(nphysical,npoints,x,y)
  condition_streamfunction.streamfunction_condition(dirichlet_pts[3],streamfunction_LHS0,neighbors_nodes)
 # ---------------------------------------------------------------------------------


 # ---------------------------------------------------------------------------------
 # Mini Element
 elif polynomial_option == 2:
  # Applying vx condition
  xvelocity_LHS0 = sps.lil_matrix.copy(M)
  condition_xvelocity = benchmark_problems.Poiseuille(nphysical,npoints,x,y)
  condition_xvelocity.neumann_condition(neumann_edges[1])
  condition_xvelocity.dirichlet_condition(dirichlet_pts[1])
  condition_xvelocity.gaussian_elimination(xvelocity_LHS0,neighbors_nodes)
  vorticity_ibc = condition_xvelocity.ibc
  benchmark_problem = condition_xvelocity.benchmark_problem
 
  # Applying vy condition
  yvelocity_LHS0 = sps.lil_matrix.copy(M)
  condition_yvelocity = benchmark_problems.Poiseuille(nphysical,npoints,x,y)
  condition_yvelocity.neumann_condition(neumann_edges[2])
  condition_yvelocity.dirichlet_condition(dirichlet_pts[2])
  condition_yvelocity.gaussian_elimination(yvelocity_LHS0,neighbors_nodes)
 
  # Applying psi condition
  streamfunction_LHS0 = sps.lil_matrix.copy(K)
  condition_streamfunction = benchmark_problems.Poiseuille(nphysical,npoints,x,y)
  condition_streamfunction.streamfunction_condition(dirichlet_pts[3],streamfunction_LHS0,neighbors_nodes)
 # ---------------------------------------------------------------------------------



 # ---------------------------------------------------------------------------------
 # Quad Element
 elif polynomial_option == 3:
  # Applying vx condition
  xvelocity_LHS0 = sps.lil_matrix.copy(M)
  condition_xvelocity = benchmark_problems.QuadPoiseuille(nphysical,npoints,x,y)
  condition_xvelocity.neumann_condition(neumann_edges[1])
  condition_xvelocity.dirichlet_condition(dirichlet_pts[1])
  condition_xvelocity.gaussian_elimination(xvelocity_LHS0,neighbors_nodes)
  vorticity_ibc = condition_xvelocity.ibc
  benchmark_problem = condition_xvelocity.benchmark_problem
 
  # Applying vy condition
  yvelocity_LHS0 = sps.lil_matrix.copy(M)
  condition_yvelocity = benchmark_problems.QuadPoiseuille(nphysical,npoints,x,y)
  condition_yvelocity.neumann_condition(neumann_edges[2])
  condition_yvelocity.dirichlet_condition(dirichlet_pts[2])
  condition_yvelocity.gaussian_elimination(yvelocity_LHS0,neighbors_nodes)
 
  # Applying psi condition
  streamfunction_LHS0 = sps.lil_matrix.copy(K)
  condition_streamfunction = benchmark_problems.QuadPoiseuille(nphysical,npoints,x,y)
  condition_streamfunction.streamfunction_condition(dirichlet_pts[3],streamfunction_LHS0,neighbors_nodes)
 # ---------------------------------------------------------------------------------





 #---------- Step 2 - Compute the boundary conditions for vorticity --------------
 vorticity_RHS = sps.lil_matrix.dot(Gx,vy) - sps.lil_matrix.dot(Gy,vx)
 vorticity_LHS = sps.lil_matrix.copy(M)
 vorticity_bc_1 = scipy.sparse.linalg.cg(vorticity_LHS,vorticity_RHS,vorticity_bc_1, maxiter=1.0e+05, tol=1.0e-05)
 vorticity_bc_1 = vorticity_bc_1[0].reshape((len(vorticity_bc_1[0]),1))


 # Gaussian elimination
 vorticity_bc_dirichlet = np.zeros([npoints,1], dtype = float)
 vorticity_bc_neumann = np.zeros([npoints,1], dtype = float)
 vorticity_bc_2 = np.ones([npoints,1], dtype = float)
 vorticity_LHS = ((np.copy(M)/dt) + (1.0/Re)*np.copy(K))
 for mm in vorticity_ibc:
  for nn in neighbors_nodes[mm]:
   vorticity_bc_dirichlet[nn] -= float(vorticity_LHS[nn,mm]*vorticity_bc_1[mm])
   vorticity_LHS[nn,mm] = 0.0
   vorticity_LHS[mm,nn] = 0.0
   
  vorticity_LHS[mm,mm] = 1.0
  vorticity_bc_dirichlet[mm] = vorticity_bc_1[mm]
  vorticity_bc_2[mm] = 0.0
 #----------------------------------------------------------------------------------



 #---------- Step 3 - Solve the vorticity transport equation ----------------------
 # Taylor Galerkin Scheme
 if scheme_option == 1:
  scheme_name = 'Taylor Galerkin'
  A = np.copy(M)/dt
  vorticity_RHS = sps.lil_matrix.dot(A,w) - np.multiply(vx,sps.lil_matrix.dot(Gx,w))\
        - np.multiply(vy,sps.lil_matrix.dot(Gy,w))\
         (dt/2.0)*np.multiply(vx,(np.multiply(vx,sps.lil_matrix.dot(Kxx,w)) + np.multiply(vy,sps.lil_matrix.dot(Kyx,w))))\
        - (dt/2.0)*np.multiply(vy,(np.multiply(vx,sps.lil_matrix.dot(Kxy,w)) + np.multiply(vy,sps.lil_matrix.dot(Kyy,w))))
  vorticity_RHS = np.multiply(vorticity_RHS,vorticity_bc_2)
  vorticity_RHS = vorticity_RHS + vorticity_bc_dirichlet
  w = scipy.sparse.linalg.cg(vorticity_LHS,vorticity_RHS,w, maxiter=1.0e+05, tol=1.0e-05)
  w = w[0].reshape((len(w[0]),1))



 # Semi-Lagrangian Scheme
 elif scheme_option == 2:

  # Linear Element   
  if polynomial_option == 1:
   scheme_name = 'Semi Lagrangian Linear'
   w_d = semi_lagrangian.Linear2D(npoints, neighbors_elements, IEN, x, y, vx_SL, vy_SL, dt, w)
   A = np.copy(M)/dt
   vorticity_RHS = sps.lil_matrix.dot(A,w_d)

   vorticity_RHS = vorticity_RHS + (1.0/Re)*vorticity_bc_neumann
   vorticity_RHS = np.multiply(vorticity_RHS,vorticity_bc_2)
   vorticity_RHS = vorticity_RHS + vorticity_bc_dirichlet

   w = scipy.sparse.linalg.cg(vorticity_LHS,vorticity_RHS,w, maxiter=1.0e+05, tol=1.0e-05)
   w = w[0].reshape((len(w[0]),1))

  # Mini Element   
  elif polynomial_option == 2:
   scheme_name = 'Semi Lagrangian Mini'
   w_d = semi_lagrangian.Mini2D(npoints, neighbors_elements, IEN, x, y, vx_SL, vy_SL, dt, w)
   A = np.copy(M)/dt
   vorticity_RHS = sps.lil_matrix.dot(A,w_d)

   vorticity_RHS = vorticity_RHS + (1.0/Re)*vorticity_bc_neumann
   vorticity_RHS = np.multiply(vorticity_RHS,vorticity_bc_2)
   vorticity_RHS = vorticity_RHS + vorticity_bc_dirichlet

   w = scipy.sparse.linalg.cg(vorticity_LHS,vorticity_RHS,w, maxiter=1.0e+05, tol=1.0e-05)
   w = w[0].reshape((len(w[0]),1))

  # Quad Element   
  elif polynomial_option == 3:
   scheme_name = 'Semi Lagrangian Quad'
   w_d = semi_lagrangian.Quad2D(npoints, neighbors_elements, IEN, x, y, vx_SL, vy_SL, dt, w)
   A = np.copy(M)/dt
   vorticity_RHS = sps.lil_matrix.dot(A,w_d)

   vorticity_RHS = vorticity_RHS + (1.0/Re)*vorticity_bc_neumann
   vorticity_RHS = np.multiply(vorticity_RHS,vorticity_bc_2)
   vorticity_RHS = vorticity_RHS + vorticity_bc_dirichlet

   w = scipy.sparse.linalg.cg(vorticity_LHS,vorticity_RHS,w, maxiter=1.0e+05, tol=1.0e-05)
   w = w[0].reshape((len(w[0]),1)) 

  else:
   print ""
   print " Error: Simulator Scheme not found"
   print ""
   sys.exit()
 #----------------------------------------------------------------------------------



 #---------- Step 4 - Solve the streamline equation --------------------------------
 # Solve Streamline
 # psi condition
 streamfunction_RHS = sps.lil_matrix.dot(M,w)
 streamfunction_RHS = np.multiply(streamfunction_RHS,condition_streamfunction.bc_2)
 streamfunction_RHS = streamfunction_RHS + condition_streamfunction.bc_dirichlet
 psi = scipy.sparse.linalg.cg(condition_streamfunction.LHS,streamfunction_RHS,psi, maxiter=1.0e+05, tol=1.0e-05)
 psi = psi[0].reshape((len(psi[0]),1))
 #----------------------------------------------------------------------------------



 #---------- Step 5 - Compute the velocity field -----------------------------------
 # Velocity vx
 xvelocity_RHS = sps.lil_matrix.dot(Gy,psi)
 xvelocity_RHS = np.multiply(xvelocity_RHS,condition_xvelocity.bc_2)
 xvelocity_RHS = xvelocity_RHS + condition_xvelocity.bc_dirichlet
 vx = scipy.sparse.linalg.cg(condition_xvelocity.LHS,xvelocity_RHS,vx, maxiter=1.0e+05, tol=1.0e-05)
 vx = vx[0].reshape((len(vx[0]),1))
 
 # Velocity vy
 yvelocity_RHS = -sps.lil_matrix.dot(Gx,psi)
 yvelocity_RHS = np.multiply(yvelocity_RHS,condition_yvelocity.bc_2)
 yvelocity_RHS = yvelocity_RHS + condition_yvelocity.bc_dirichlet
 vy = scipy.sparse.linalg.cg(condition_yvelocity.LHS,yvelocity_RHS,vy, maxiter=1.0e+05, tol=1.0e-05)
 vy = vy[0].reshape((len(vy[0]),1))
 #----------------------------------------------------------------------------------


end_time = time()
solution_time = end_time - start_time
print ' time duration: %.1f seconds' %solution_time
print ""





print ' ----------------'
print ' SAVING RELATORY:'
print ' ----------------'
print ""
print ' End simulation. Relatory saved in %s' %directory_save
print ""

# -------------------------------- Export Relatory ---------------------------------------
relatory.export(save.path, directory_save, sys.argv[0], benchmark_problem, scheme_name, mesh_name, equation_number, npoints, nelem, length_min, dt, nt, Re, Sc, import_mesh_time, assembly_time, bc_apply_time, solution_time, polynomial_order, gausspoints)
