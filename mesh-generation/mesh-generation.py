#! /usr/bin/env python3

import pyvista as pv
import numpy as np
#from pyFTLE2 import compute_ftle
import matplotlib.pyplot as plt
import time
import sys as sys
from multiprocess import Pool
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import shared_memory
from scipy.spatial import cKDTree as KDTree

import argparse
from ctypes import *

from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp

def compute_velocity_vector_2D(timeVector, points):
    velocity = []
    for i in range(len(timeVector)):
        for j in range(len(points)):
            velocity.append(compute_velocity_2D(timeVector[i], points[j][0], points[j][1]))
    return velocity

def compute_velocity_2D(t, x, y):
    A = 0.1
    omega = 2*np.pi/10
    epsilon = 0.25
    a_t = epsilon*np.sin(omega*t)
    b_t = 1 - 2*epsilon*np.sin(omega*t)
    f = a_t*x**2+b_t*x
    dfdx = 2*a_t*x+b_t
    u = -A*np.pi*np.sin(np.pi*f)*np.cos(np.pi*y)
    v = np.pi * A*np.cos(np.pi*f)*np.sin(np.pi*y)*dfdx
    return u, v

def compute_flowmap_2D(x_, y_, t_, vx, vy, mesh, gridpoints, timeVector, odemethod='RK45', nDim=2, number_of_threads=4):
    print('Building interpolator...')
    # RegularGridInterpolator
    Rinterp_x = RegularGridInterpolator(
        (x_, y_, t_), vx, bounds_error=False, fill_value=0, method='linear')
    Rinterp_y = RegularGridInterpolator(
        (x_, y_, t_), vy, bounds_error=False, fill_value=0, method='linear')

    def interpolator(pts):
        return np.vstack((np.array(Rinterp_x(pts)), np.array(Rinterp_y(pts)))).T

    print('Interpolante constructed.')

    def trajectory(t, Z):
        number_of_vectors = int(Z.size / nDim)
        space_vector = np.reshape(Z, (number_of_vectors, nDim))
        time_vector = np.ones(int(space_vector.size / nDim))*t
        X = np.vstack((space_vector.T, time_vector)).T
        out = interpolator(X)
        out = np.ravel(out)
        return out

    number_of_odes = int(gridpoints.size)
    print('ODEs: ', number_of_odes)

    print('Solving...')

    c0 = np.ravel(gridpoints)

    def wrap_ode(c0):
        sol = solve_ivp(trajectory, timeVector, c0, t_eval=[
                        timeVector[1]], method=odemethod)
        return sol.y

    t_ini = time.time()
    solution = wrap_ode(c0)
    t_fin = time.time()

    print('Solved in %4.2f seconds' % (t_fin - t_ini))
    # ------------------ FINISH PARALLEL REGION  -----------------------

    solution = np.ravel(np.vstack(solution))
    solution = np.reshape(solution, (int(solution.size/nDim), nDim))

    if nDim == 2:
        mesh['flowmapZ'] = np.zeros_like(solution[:, 0])
    else:
        mesh['flowmapZ'] = solution[:, 2]

    mesh['flowmapX'] = solution[:, 0]
    mesh['flowmapY'] = solution[:, 1]
    return mesh

def compute_velocity_3D(t, x, y, z):
    '''
        returns the velocity field of an ABC flow
    '''
    A = np.sqrt(3)
    B = np.sqrt(2)
    C = 1
    omega = 2*np.pi/10
    epsilon = 0.1
    A_t = A+epsilon*np.cos(omega*t)
    u = A_t*np.sin(z) + C*np.cos(y)
    v = B*np.sin(x) + A_t*np.cos(z)
    w = C*np.sin(y) + B*np.cos(x)

    return u, v, w

def compute_velocity_vector_3D(timeVector, points):
    velocity = []
    for i in range(len(timeVector)):
        for j in range(len(points)):
            velocity.append(compute_velocity_3D(
                timeVector[i], points[j][0], points[j][1], points[j][2]))
    return velocity

def compute_flowmap_3D(x_, y_, z_, t_, vx, vy, vz, mesh, gridpoints, timeVector, odemethod='RK45', nDim=3, number_of_threads=4):

    print('Building interpolator...')

    # RegularGridInterpolator
    Rinterp_x = RegularGridInterpolator(
        (x_, y_, z_, t_), vx, bounds_error=False, fill_value=0, method='linear')
    Rinterp_y = RegularGridInterpolator(
        (x_, y_, z_, t_), vy, bounds_error=False, fill_value=0, method='linear')
    Rinterp_z = RegularGridInterpolator(
        (x_, y_, z_, t_), vz, bounds_error=False, fill_value=0, method='linear')

    def interpolator(pts):
        return np.vstack((np.array(Rinterp_x(pts)), np.array(Rinterp_y(pts)), np.array(Rinterp_z(pts)))).T

    print('Interpolante constructed.')

    def trajectory(t, Z):
        number_of_vectors = int(Z.size / nDim)
        space_vector = np.reshape(Z, (number_of_vectors, nDim))
        time_vector = np.ones(int(space_vector.size / nDim))*t
        X = np.vstack((space_vector.T, time_vector)).T
        out = interpolator(X)
        out = np.ravel(out)
        return out

    number_of_odes = int(gridpoints.size)
    print('ODEs: ', number_of_odes)

    print('Solving...')
    p = Pool(number_of_threads) #RCS

    #results = Parallel(n_jobs=number_of_threads)(delayed(define_tetrahedra)(i) for i in range(n_cells))

    points_splitted = np.array_split(gridpoints, number_of_threads) #RCS
    c0 = [] #RCS
    for i, points_array in enumerate(points_splitted): #RCS
        c0.append(np.ravel(points_array)) #RCS


    ###RCS c0 = np.ravel(gridpoints)

    def wrap_ode(c0):
        sol = solve_ivp(trajectory, timeVector, c0, t_eval=[
                        timeVector[1]], method=odemethod)
        return sol.y

    t_ini = time.time()
    ###RCS solution = wrap_ode(c0)
    parallel_solution = p.map(wrap_ode, c0) #RCS
    t_fin = time.time()

    print('Solved in %4.2f seconds' % (t_fin - t_ini))
    # ------------------ FINISH PARALLEL REGION  -----------------------

    ###RCS solution = np.ravel(np.vstack(solution))
    solution = np.ravel(np.vstack(parallel_solution))
    solution = np.reshape(solution, (int(solution.size/nDim), nDim))

    if nDim == 2:
        mesh['flowmapZ'] = np.zeros_like(solution[:, 0])
    else:
        mesh['flowmapZ'] = solution[:, 2]

    mesh['flowmapX'] = solution[:, 0]
    mesh['flowmapY'] = solution[:, 1]
    return mesh

def main_2D (args):
    
    t_start = time.time()

    # Parse args
    nDim = int(args.nDim[0])
    ntimes_eval = 1
    t0_eval = float(args.t0_eval)
    tdelta_eval= 1.0
    coords_file = args.coords_file
    faces_file = args.faces_file
    times_file = args.times_file
    vel_file = args.vel_file
    nsteps_rk4 = int(args.nsteps_rk4)
    sched_policy = -1
    if args.sched_policy == "SEQUENTIAL":
        sched_policy = 1
    elif args.sched_policy == "OMP_STATIC":
        sched_policy = 2
    elif args.sched_policy == "OMP_DYNAMIC":
        sched_policy = 3
    elif args.sched_policy == "OMP_GUIDED":
        sched_policy = 4
    chunk_size = int(args.chunk_size)

    # Output configuration for compute flowmap
    print2file = 1
    output_file = args.out_file

    print("x_steps_axis -> %d" % int(args.x_steps_axis))
    print("y_steps_axis -> %d" % int(args.y_steps_axis))

    x_ = np.linspace(0, 2, int(args.x_steps_axis))
    y_ = np.linspace(0, 1, int(args.y_steps_axis))
    t_ = np.linspace(0, 10, 100)
    xx, yy, zz = np.meshgrid(x_, y_, [0])

    points = np.column_stack((xx.ravel(order="F"),
                            yy.ravel(order="F"),
                            zz.ravel(order="F")))
    print("Number of points: %d" % len(points), flush=True)

    cloud = pv.PolyData(points)
    print("Constructing 2D Delaunay triangulation of the mesh...", flush=True)
    mesh = cloud.delaunay_2d()

    xg, yg, tg = np.meshgrid(x_, y_, t_, indexing='ij')
    _v = np.array(compute_velocity_2D(tg, xg, yg))
    vx = _v[0]
    vy = _v[1]

    gridpoints = np.array(mesh.points)
    vel = compute_velocity_vector_2D(t_, gridpoints)

    gridpoints = np.delete(gridpoints, 2, 1)

    # RCS -- Store velocity, coords, time in files for C code
    t_preprocess = time.time()
    print("Python data generated: "+str(t_preprocess-t_start), flush=True)

    fc = open(coords_file, "w")
    ff = open(faces_file, "w")
    ft = open(times_file, "w")
    fv = open(vel_file, "w")

    fc.write(str(len(points))+'\n')
    for index in range(len(points)):
        fc.write(str(points[index][0])+'\n')
        fc.write(str(points[index][1])+'\n')

    ft.write(str(len(t_))+'\n')
    for index in range(len(t_)):
       ft.write(str(t_[index])+'\n')
       for jindex in range(len(points)):
           fv.write(str(vel[index*len(points)+jindex][0])+'\n')
           fv.write(str(vel[index*len(points)+jindex][1])+'\n')

    faces = mesh.faces.reshape((-1,4))[:, 1:4]
    print("Number of faces: %d" % len(faces) , flush=True )
    ff.write(str(len(faces))+'\n')    
    for index in range(len(faces)):
        ff.write(str(faces[index][0])+'\n')
        ff.write(str(faces[index][1])+'\n')
        ff.write(str(faces[index][2])+'\n')

    # RCS -- Close files
    fc.close()
    ff.close()
    ft.close()
    fv.close()

    # End RCS
    t_store = time.time()
    print("Python data saved in files: "+str(t_store-t_preprocess), flush=True)

    print("Starting compute flowmap...", flush=True)
    flowmap = compute_flowmap_2D(x_, y_, t_, vx, vy, mesh, gridpoints, [0, 8])

    truefm_x = mesh['flowmapX']
    truefm_y = mesh['flowmapY']

    ff = open(output_file, "w")
    for index in range(len(points)):
        ff.write(str(truefm_x[index])+'\n')
        ff.write(str(truefm_y[index])+'\n')
    ff.close()

    t_end = time.time()
    print("Total time elapsed: "+str(t_end-t_start), flush=True)

def main_3D (args):

    t_start = time.time()

    # Parse args
    nDim = int(args.nDim[0])
    ntimes_eval = 1
    t0_eval = float(args.t0_eval)
    tdelta_eval= 1.0
    coords_file = args.coords_file
    faces_file = args.faces_file
    times_file = args.times_file
    vel_file = args.vel_file
    nsteps_rk4 = int(args.nsteps_rk4)
    sched_policy = -1
    if args.sched_policy == "SEQUENTIAL":
        sched_policy = 1
    elif args.sched_policy == "OMP_STATIC":
        sched_policy = 2
    elif args.sched_policy == "OMP_DYNAMIC":
        sched_policy = 3
    elif args.sched_policy == "OMP_GUIDED":
        sched_policy = 4
    chunk_size = int(args.chunk_size)

    # Output configuration for compute flowmap
    print2file = 1
    output_file = args.out_file

    print("x_steps_axis -> %d" % int(args.x_steps_axis))
    print("y_steps_axis -> %d" % int(args.y_steps_axis))
    print("z_steps_axis -> %d" % int(args.z_steps_axis))

    x_ = np.linspace(0, 1, int(args.x_steps_axis))
    y_ = np.linspace(0, 1, int(args.y_steps_axis))
    z_ = np.linspace(0, 1, int(args.z_steps_axis))
    t_ = np.linspace(0, 10, 20)
    xx, yy, zz = np.meshgrid(x_, y_, z_)

    points = np.column_stack((xx.ravel(order="F"),
                            yy.ravel(order="F"),
                            zz.ravel(order="F")))

    print("Number of points: %d" % len(points), flush=True)
    cloud = pv.PolyData(points)
    print("Constructing 3D Delaunay triangulation of the mesh...", flush=True)
    mesh = cloud.delaunay_3d()
    mesh = mesh.point_data_to_cell_data()
    #test = mesh.poin
    #mesh.plot(show_edges=True)
    n_cells = mesh.n_cells
    #print(type(mesh))
    #shm = shared_memory.SharedMemory(name='shared_mesh', create=True, size=mesh.nbytes)
    
    data = np.empty((n_cells, 4))
    shm = shared_memory.SharedMemory(name='shared_tetrahedra3', create=True, size=data.nbytes)
    tetrahedra = np.ndarray(data.shape, dtype=str(data.dtype), buffer=shm.buf)
    # t = data
    
    print("Generating KDTree...", flush=True)
    # KDTree
    kdtree_mesh_points = KDTree(mesh.points)

    # Ensure mesh and data exists
    def define_tetrahedra(p):
        ini = p[0]
        fin = p [1]
        t_local = []
        for i in range(fin - ini):
            #print(t_local, flush=True)
            cell = mesh.extract_cells(ini + i)
            # Get shared memory
            #t_local.append ( [ mesh.find_closest_point(cell.points[0]), mesh.find_closest_point(cell.points[1]), mesh.find_closest_point(cell.points[2]), mesh.find_closest_point(cell.points[3]) ] ) 
            distances, indices = kdtree_mesh_points.query(cell.points)
            t_local.append(indices)

        #print(t_local, flush=True)
        existing_shm = shared_memory.SharedMemory(name='shared_tetrahedra3')
        t = np.ndarray(data.shape, dtype=str(data.dtype), buffer=existing_shm.buf)
        #Find index of closest point in this mesh to the given point.
        for i in range(fin - ini):
            i_ = ini+i
            t[i_][0] = t_local[i][0]
            t[i_][1] = t_local[i][1]
            t[i_][2] = t_local[i][2]
            t[i_][3] = t_local[i][3]
        existing_shm.close()
        #return (mesh.find_closest_point(cell.points[0]), mesh.find_closest_point(cell.points[1]), mesh.find_closest_point(cell.points[2]), mesh.find_closest_point(cell.points[0]))

    print("Generating tetrahedra...", flush=True)

    # Create tetrahedra in parallel
    num_cores = int(args.num_cores) # multiprocessing.cpu_count()
    #num_cores = 4
    # DEFAULT results = Parallel(n_jobs=num_cores, prefer=None)(delayed(define_tetrahedra)(i) for i in range(n_cells))  
    # results = Parallel(n_jobs=num_cores, prefer="processes")(delayed(define_tetrahedra)(i) for i in range(n_cells))
    # results = Parallel(n_jobs=num_cores, prefer="threads")(delayed(define_tetrahedra)(i) for i in range(n_cells))
    #results = Parallel(n_jobs=num_cores)(delayed(define_tetrahedra)(i) for i in range(n_cells))

    step = n_cells // num_cores
    start = 0
    stop = step
    parts = []
    #print("n_cells: %d" % n_cells)
    for i in range(num_cores):
        #print("ini: %d, stop: %d" % (start, stop), flush=True)
        #print(a[start:stop])
        parts.append( (start, stop) )
        start += step
        stop += step
        #last core
        if i+1 == (num_cores-1):
            stop += n_cells % num_cores

    results = Parallel(n_jobs=num_cores)(delayed(define_tetrahedra)(p) for p in parts)

    # tetrahedra = np.empty((n_cells, 4))
    # for i in range(len(results)):
    #     tetrahedra[i][0] = results[i][0]
    #     tetrahedra[i][1] = results[i][1]
    #     tetrahedra[i][2] = results[i][2]
    #     tetrahedra[i][3] = results[i][3]

    print("Number of tetrahedra: %d" % len(tetrahedra) , flush=True )
    '''
    for i in range(n_cells):
        cell = mesh.extract_cells(i)
        # Find index of closest point in this mesh to the given point.
        tetrahedra[i][0] = mesh.find_closest_point(cell.points[0])
        tetrahedra[i][1] = mesh.find_closest_point(cell.points[1])
        tetrahedra[i][2] = mesh.find_closest_point(cell.points[2])
        tetrahedra[i][3] = mesh.find_closest_point(cell.points[3])
    '''
    xg, yg, zg, tg = np.meshgrid(x_, y_, z_, t_, indexing='ij')
    _v = np.array(compute_velocity_3D(tg, xg, yg, zg) )
    vx = _v[0]
    vy = _v[1]
    vz = _v[2]

    gridpoints = np.array(mesh.points)
    vel = compute_velocity_vector_3D(t_, gridpoints)

    #gridpoints = np.delete(gridpoints, 2, 1)

    t_preprocess = time.time()
    print("Python data generated: "+str(t_preprocess-t_start), flush=True)

    # RCS -- Store velocity, coords, time in files for C code

    fc = open(coords_file, "w")
    ff = open(faces_file, "w")
    ft = open(times_file, "w")
    fv = open(vel_file, "w")

    fc.write(str(len(points))+'\n')
    for index in range(len(points)):
        fc.write( str(points[index][0])+'\n'+ str(points[index][1])+'\n' + str(points[index][2])+'\n')

    ft.write(str(len(t_))+'\n')
    for index in range(len(t_)):
       ft.write(str(t_[index])+'\n')
       for jindex in range(len(points)):
           fv.write(str(vel[index*len(points)+jindex][0])+'\n'+str(vel[index*len(points)+jindex][1])+'\n'+str(vel[index*len(points)+jindex][2])+'\n')

    ff.write(str(len(tetrahedra))+'\n')
    for index in range(len(tetrahedra)):
        ff.write(str(tetrahedra[index][0])+'\n' + str(tetrahedra[index][1])+'\n' + str(tetrahedra[index][2])+'\n' + str(tetrahedra[index][3])+'\n')

    # Shared memory end
    shm.close()
    shm.unlink()

    # RCS -- Close files
    fc.close()
    ff.close()
    ft.close()
    fv.close()

    # End RCS
    t_store = time.time()
    print("Python data saved in files: "+str(t_store-t_preprocess), flush=True)

    print("Starting compute flowmap...", flush=True)
    flowmap = compute_flowmap_3D(x_, y_, z_, t_, vx, vy, vz, mesh, gridpoints, [0, 8], number_of_threads=num_cores)

    truefm_x = mesh['flowmapX']
    truefm_y = mesh['flowmapY']
    truefm_z = mesh['flowmapZ']

    ff = open(output_file, "w")
    for index in range(len(points)):
        ff.write(str(truefm_x[index])+'\n')
        ff.write(str(truefm_y[index])+'\n')
        ff.write(str(truefm_z[index])+'\n')
    ff.close()
    
    '''
    COMPUTE_FLOWMAP_C_FUNCTIONS.compute_flowmap(c_int(nDim), c_int(ntimes_eval), c_double(t0_eval), c_double(tdelta_eval), c_char_p(coords_file.encode('utf-8')), c_char_p(faces_file.encode('utf-8')), c_char_p(times_file.encode('utf-8')), c_char_p(vel_file.encode('utf-8')), c_int(nsteps_rk4), c_int(sched_policy), c_int(chunk_size), c_int(print2file), c_char_p(output_file.encode('utf-8')) )
    '''

    t_end = time.time()
    print("Total time elapsed: "+str(t_end-t_start), flush=True)


if __name__ == '__main__':

    #COMPUTE_FLOWMAP_C_FUNCTIONS = CDLL(SHARED_LIBRARY)

    NDIM_CHOICES = ["2D","3D"]
    NDIM_HELP = "Dimensions of the space. Choices: %s" % (", ".join(NDIM_CHOICES) )
    COMPUTE_FLOWMAP_POLICIES = ["SEQUENTIAL", "OMP_STATIC", "OMP_DYNAMIC", "OMP_GUIDED"]
    COMPUTE_FLOWMAP_POLICIES_HELP = "Scheduling policy. Choices: %s" % (", ".join(COMPUTE_FLOWMAP_POLICIES))


    parser = argparse.ArgumentParser()
    parser.add_argument(metavar="<nDim>", dest="nDim", help=NDIM_HELP, choices=NDIM_CHOICES, default="3D")
    # parser.add_argument(metavar="<ntimes_eval>", dest="ntimes_eval", help="Quantity of t values to choose (starting at t0 and adding tdelta each time) which we want to evaluate")
    parser.add_argument(metavar="<t0_eval>", dest="t0_eval", help="First t to evaluate")
    # parser.add_argument(metavar="<tdelta_eval>", dest="tdelta_eval", help="t increment in each t to evaluate after t0")
    parser.add_argument(metavar="<coords_file>", dest="coords_file", help="File path where mesh coordinates will be stored", default="coords.txt")
    parser.add_argument(metavar="<faces_file>", dest="faces_file", help="File path where mesh faces will be stored", default="faces.txt")
    parser.add_argument(metavar="<times_file>", dest="times_file", help="File path where time data will be stored", default="times.txt")
    parser.add_argument(metavar="<vel_file>", dest="vel_file", help="File path where velocity data will be stored", default="vel.txt")
    parser.add_argument(metavar="<nsteps_rk4>", dest="nsteps_rk4", help="Number of iterations to perform in the RK4 call")
    parser.add_argument(metavar="<sched_policy>", dest="sched_policy", help=COMPUTE_FLOWMAP_POLICIES_HELP, choices=COMPUTE_FLOWMAP_POLICIES)
    parser.add_argument(metavar="<chunk_size>", dest="chunk_size", help="Size of the chunk for the chosen scheduling policy")
    parser.add_argument(metavar="<output_file>", dest="out_file", help="File path where flowmap data will be stored", default="flowmap.txt")
    parser.add_argument(metavar="<num_cores>", dest="num_cores", help="Threads to use", type=int)
    parser.add_argument(metavar="<x_steps_axis>", dest="x_steps_axis", help="Steps in X axis (for linspace)",  default=-1, type=int )
    parser.add_argument(metavar="<y_steps_axis>", dest="y_steps_axis", help="Steps in Y axis (for linspace)", default=-1, type=int )
    parser.add_argument(metavar="<z_steps_axis>", dest="z_steps_axis", help="Steps in Z axis (for linspace)", default=-1, nargs='?', type=int)


    args = parser.parse_args()
    
    # Ensure z_steps_axis is defined for 3D
    if args.x_steps_axis <= 0 or args.y_steps_axis <= 0:
        print("ERROR: <x_steps_axis> and <y_steps_axis> must be positive integers")
        sys.exit()
    if args.nDim == "3D" and args.z_steps_axis == -1:
        print("ERROR: you must define <z_steps_axis> for 3D")
        parser.print_help()
        sys.exit()

    
    eval("main_"+args.nDim)(args)


    
