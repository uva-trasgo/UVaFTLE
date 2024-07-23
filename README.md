<p align="center">
    <img src="UVaFTLE_Logo.png" alt="UvaFTLE logo">
</p>

# UVaFTLE: Lagrangian finite time Lyapunov exponent extraction for fluid dynamic applications

UVaFTLE is an open source C implementation for determining Lagrangian Coherent Structures
(LCS). UVaFTLE is also a parallel implementation, using OpenMP, CUDA, HIP, OpenCL, and SYCL for
solving the problem in shared-memory multiprocessors, NVIDIA GPUS, AMD GPUS, Intel FPGAs
(experimental), and heterogeneous systems, respectively.

## Compiling UVaFTLE

Package dependencies: 

* cmake (>= 3.8)
* OpenMP
* Cuda Toolkit 
* ROCm Toolkit
* SYCL compiler (tested with OneAPI 2024.1.0 and AdaptiveCpp 24.02.0)

Once the package has been obtained, simply run:

```bash
$ cd UVaFTLE
$ cmake . -DCMAKE_INSTALL_PREFIX="installation_folder"  -DUVAFTLE_OPTIONS=yes/no
$ make
$ make install
```
The possible options to configure UVaFTLE are:


* *-DWITH_OMP*: Enables the compilation of the OpenMP version
* *-DWITH_CUDA*: Enables the compilation of the CUDA version
* *-DWITH_ROCM*: Enables the compilation of the ROCm/HIP version
* *-DWITH_SYCL_OMP*: Enables the compilation of the SYCL version using the OpenMP backend of AdaptiveCpp
* *-DWITH_SYCL_CUDA*: Enables the compilation of the SYCL version using the CUDA backend of AdaptiveCpp
* *-DWITH_SYCL_ROCM*: Enables the compilation of the SYCL version using the HIP backend of AdaptiveCpp
* *-DWITH_SYCL_GENERIC*: Enables the compilation of the SYCL version using the just-in-time compiler of AdaptiveCpp
* *-DWITH_ALL_VERSIONS*: Enables the compilation of all UvaFTLE versions 

Take into account that:

* If the installation folder is not set using -DCMAKE_INSTALL_PREFIX,  the executables are installed in	*UVaFTLE/bin*. 
* If your version of CMake is 3.18 or greater, you should specify the cuda  arch using the option  "-DCUDA_ARCH=arch_code". In other case, the arch 75 is selected by default.  In order to identify your architecture, you could use [this quick quide](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/).
* If the compilation of CUDA of HIP backend using AdaptiveCpp is enabled, you must specify the target architecture in the CMakeList.txt (search for --acpp-targets flag).
* There are specific scripts in *SYCL* and *SYCL-usm* folders for compiling the SYCL code using OneAPI DPC++.

If you have an older version of cmake, UVaFTLE can be compiled using the
makefiles, but the *bin* directories must be created explicitly:


```bash
$ cd UVaFTLE
$ mkdir CPU/bin CUDA/bin  
$ cd CPU
$ make
$ cd CUDA
$ make 
```

## Running UVaFTLE

### Running CUDA version

Before running UVaFTLE using CUDA, you must specify the number of GPUS that will solve the problem using the environment variable *OMP_NUM_THREADS*. For example, 
to run UVaFTLE using two GPUS, you must type:

```bash
$ export OMP_NUM_THREADS=2
```
After that, you can run UVaFTLE running this command:

```bash
$ ftle_cuda <nDim> <coords_file> <faces_file> <flowmap_file> <t_eval> <print2file>
```
where: 

* *nDim* indicates the dimensions of the space (2D/3D).
* *coords_file* indicates the file where mesh coordinates are stored.
* *faces_file* indicates the file where mesh faces are stored.
* *flowmap_file* indicates the file where flowmap values are stored.
* *t_eval* indicates the time when compute ftle is desired.
* *print2file* indicates if the result is stored in output file if csv format (0: no, 1: yes). By default, the file is called *result_FTLE.csv* and it is stored in the current directory. 

### Running shared-memory versions

In this case,  the environment variable *OMP_NUM_THREADS* is used to specify the number of OpenMP threads. For example, 
to run UVaFTLE using 16 threads, you must type:

```bash
$ export OMP_NUM_THREADS=16
```
After that, you can run UVaFTLE running this command:

```bash
$ ftle_sched <nDim> <coords_file> <faces_file> <flowmap_file> <t_eval> <nth> <print2file>
```
where: 

* **ftle_sched** indicates the UVAFTLE implementation chosen. There are three different versions for running UVaFTLE, varying the OpenMP scheduling: *ftle_static*, *ftle_dynamic* and *ftle_guided*.
* *nDim* indicates the dimensions of the space (2D/3D).
* *coords_file* indicates the file where mesh coordinates are stored.
* *faces_file* indicates the file where mesh faces are stored.
* *flowmap_file* indicates the file where flowmap values are stored.
* *t_eval* indicates the time when compute ftle is desired.
* *nth* indicates the number of OpenMP threads to use.
* *print2file* indicates if the result is stored in output file if csv format (0: no, 1: yes). By default, the file is called *result_FTLE.csv* and it is stored in the current directory. 

## Citation

If you write a scientific paper describing research that makes substantive use of
UVaFTLE , we would appreciate that you cite the following paper:

* [UVaFTLE: Lagrangian finite time Lyapunov exponent extraction for fluid dynamic applications](https://link.springer.com/article/10.1007/s11227-022-05017-x) 

```BibTeX
	@Article{Carratala2023:UvaFTLE,
	author = "Carratal{\'a}-S{\'a}ez, Roc{\'i}o and Torres, Yuri and Sierra-Pallares, Jos{\'e} and others",
	title="UVaFTLE: Lagrangian finite time Lyapunov exponent extraction for fluid dynamic applications",
	journal="The Journal of Supercomputing",
	year="2023",
	issn="1573-0484",
	doi="https://doi.org/10.1007/s11227-022-05017-x",
	}   
```
