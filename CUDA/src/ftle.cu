#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>

#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <assert.h>

#include "ftle.h"
#include "arithmetic.h"
#include "preprocess.h"

#define blockSize 512
//#define currentGPU 0




int main(int argc, char *argv[]) {

	// Check usage
	if (argc != 7)
	{
		printf("USAGE: ./executable <nDim> <coords_file> <faces_file> <flowmap_file> <t_eval>\n");
		printf("\texecutable:    compute_ftle\n");
		printf("\tnDim:    dimensions of the space (2D/3D)\n");
		printf("\tcoords_file:   file where mesh coordinates are stored.\n");
		printf("\tfaces_file:    file where mesh faces are stored.\n");
		printf("\tflowmap_file:  file where flowmap values are stored.\n");
		printf("\tt_eval:        time when compute ftle is desired.\n");
        printf("\trepeticiones:         number of repeticiones\n");
		return 1;
	}

    /*
    int nDevices ;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (MHz): %d\n",
               prop.memoryClockRate/1024);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
        printf("  Total global memory (Gbytes) %.1f\n",(float)(prop.totalGlobalMem)/1024.0/1024.0/1024.0);
        printf("  Shared memory per block (Kbytes) %.1f\n",(float)(prop.sharedMemPerBlock)/1024.0);
        printf("  minor-major: %d-%d\n", prop.minor, prop.major);
        printf("  Warp-size: %d\n", prop.warpSize);
        printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
        printf("  Concurrent computation/communication: %s\n\n",prop.deviceOverlap ? "yes" : "no");
      }*/


    /* Initialize mesh original information */
    int nDim, nVertsPerFace, nPoints, nFaces;
    nDim = atoi(argv[1]);
    if ( nDim == 2 )
    nVertsPerFace = 3;    // 2D: faces are triangles
    else {
            if ( nDim == 3)
        nVertsPerFace = 4; // 3D: faces (volumes) are tetrahedrons
            else
    {
        printf("Wrong dimension provided (2 or 3 supported)\n");
        return 1;
    }
    }


   /* Read coordinates, faces and flowmap from Python-generated files and generate corresponding GPU vectors */
   int  check_EOF;
   char buffer[255];
   FILE *file;

   // Open file
   file = fopen( argv[2], "r" );

   // Get nPoints
   check_EOF = fscanf(file, "%s", buffer);
   if ( check_EOF == EOF )
   {
           fprintf( stderr, "Error: Unexpected EOF in read_coordinates\n" );
           exit(-1);
   }
   nPoints = atoi(buffer);
   fclose(file);

   // Get nFaces
   file = fopen( argv[3], "r" );
   check_EOF = fscanf(file, "%s", buffer);
   if ( check_EOF == EOF )
   {
           fprintf( stderr, "Error: Unexpected EOF in read_faces\n" );
           exit(-1);
   }
   nFaces = atoi(buffer);


    double t_eval =  atof(argv[5]);
       
    double times[10];
    int numThreads=0;
	double *coords; // *d_coords;
	double *flowmap;//, *d_flowmap;
	int    *faces, *d2_faces;
    double *logSqrt;
	
	int    *nFacesPerPoint, *d2_nFacesPerPoint;
	int    *facesPerPoint,  *d2_facesPerPoint;

	 int m   = nDim;
	 int lda = nDim;

	// Read coordinates, faces and flowmap values
	printf("Reading mesh points coordinates...                     ");
	coords = (double *) malloc ( sizeof(double) * nPoints * nDim );
	read_coordinates(argv[2], nDim, nPoints, coords); 
	//for ( int ii=0; ii < nPoints; ii++ ) printf("p %d coords %f %f\n", ii, coords[ii * nDim], coords[ii * nDim + 1]); 
	printf("DONE\n");
   	
	printf("Reading mesh faces vertices...                         "); 
	faces = (int *) malloc ( sizeof(int) * nFaces * nVertsPerFace );
	read_faces(argv[3], nDim, nVertsPerFace, nFaces, faces); 
	//for ( int ii=0; ii < nFaces; ii++ ) printf("f %d vtx %d %d %d\n", ii, faces[ii * nVertsPerFace], faces[ii * nVertsPerFace + 1], faces[ii * nVertsPerFace + 2]); 
	printf("DONE\n");
	
	printf("Reading mesh flowmap (x, y[, z])...                    ");
	flowmap = (double*) malloc( sizeof(double) * nPoints * nDim ); 
	read_flowmap ( argv[4], nDim, nPoints, flowmap );
	printf("DONE\n"); 
	
    int numRepes = atof(argv[6]);

    /* Allocate additional memory at the CPU */    
    logSqrt        = (double*) malloc( sizeof(double) * nPoints);       
    nFacesPerPoint = (int *) malloc( sizeof(int) * nPoints ); /* REMARK: nFacesPerPoint accumulates previous nFacesPerPoint */
   
    
	//printf("DONE\n"); fflush(stdout);	

	/* Assign faces to vertices and generate nFacesPerPoint and facesPerPoint GPU vectors */
	//printf("Setting up relationships between faces and vertices... "); //fflush(stdout);
	create_nFacesPerPoint_vector ( nDim, nPoints, nFaces, nVertsPerFace, faces, nFacesPerPoint );
    printf("Punto de contrl 1.......................................\n"); fflush(stdout);
	//for ( int ii=0; ii < nPoints; ii++ ) printf("p %d numfaces %d\n", ii, nFacesPerPoint[ii]); 
    	facesPerPoint = (int *) malloc( sizeof(int) * nFacesPerPoint[ nPoints - 1 ] );

        cudaMalloc( &d2_facesPerPoint, sizeof(int)    *   nFacesPerPoint[ nPoints - 1 ]); 
        cudaMalloc( &d2_faces,   sizeof(int)    * nFaces  * nVertsPerFace ); 
        cudaMalloc( &d2_nFacesPerPoint, sizeof(int)    * nPoints); 
        cudaMemcpy( d2_faces,   faces,   sizeof(int)    * nFaces  * nVertsPerFace, cudaMemcpyHostToDevice ); 
        cudaMemcpy( d2_nFacesPerPoint, nFacesPerPoint, sizeof(int) * nPoints,                       cudaMemcpyHostToDevice );  
	//create_facesPerPoint_vector ( nDim, nPoints, nFaces, nVertsPerFace, faces, nFacesPerPoint, facesPerPoint );
    create_facesPerPoint_vector_GPU<<<(ceil(    (double)nPoints/(double)blockSize)  +1),blockSize>>> ( nDim, nPoints, nFaces, nVertsPerFace, 
                                                                                                                    d2_faces, d2_nFacesPerPoint, d2_facesPerPoint );
    cudaMemcpy( facesPerPoint,   d2_facesPerPoint,   sizeof(int)    * nFacesPerPoint[ nPoints - 1 ], cudaMemcpyDeviceToHost ); 
    printf("Punto de contrl 2.......................................\n"); fflush(stdout);


    //printf("CHECK............ 1\n"); fflush(stdout);

    #pragma omp parallel default(none)  shared(logSqrt, nDim, nPoints, nFaces, nVertsPerFace, numRepes, numThreads, times, faces, coords, nFacesPerPoint, facesPerPoint, flowmap, t_eval, lda, m)   //shared(sched_chunk_size, t_eval, npoints, nteval, result, mesh, d_cuda_coords_vector, d_cuda_coords_vector2, d_cuda_velocity_vector, d_cuda_velocity_vector2, d_cuda_faces_vector, d_cuda_faces_vector2, d_cuda_times_vector, d_cuda_times_vector2, nsteps_rk4, numBlocks) private(ip, it, itprev) firstprivate(multigpu)
    {

        numThreads= omp_get_num_threads();
        //printf("SOY el HILO ....................%d \n",omp_get_thread_num());
       // fflush(stdout);
        cudaError_t error;
        //printf(".........................%d\n", omp_get_num_threads());
        double *d_w;
        double *d_logSqrt;
    
        double *d_A_ei, *d_W_ei; // , *d_work;
        int     *devInfo; // lwork_ei,
    
        //double *ftl_matrix;
        double *d_ftl_matrix;
    
        double *d_gra1;
        double *d_gra2;
        double *d_gra3;
        
        double *d_res;
        
        double *d_cg_tensor1;
        double *d_cg_tensor2;
        double *d_cg_tensor3;

        
        double   *d_coords;
        double  *d_flowmap;
        int       *d_faces;
        int     *d_nFacesPerPoint;
        int      *d_facesPerPoint;
        
    
    cudaSetDevice(omp_get_thread_num());

	/* Allocate memory for read data at the GPU (coords, faces, flowmap) */
	//printf("Allocating memory at the GPU and copying data to it... ");
	cudaMalloc( &d_coords,  sizeof(double) * nPoints * nDim );
	cudaMalloc( &d_faces,   sizeof(int)    * nFaces  * nVertsPerFace );
	cudaMalloc( &d_flowmap, sizeof(double) * nPoints * nDim );

	/* Copy read data to GPU (coords, faces, flowmap) */
	cudaMemcpy( d_coords,  coords,  sizeof(double) * nPoints * nDim,          cudaMemcpyHostToDevice );
	cudaMemcpy( d_faces,   faces,   sizeof(int)    * nFaces  * nVertsPerFace, cudaMemcpyHostToDevice );
	cudaMemcpy( d_flowmap, flowmap, sizeof(double) * nPoints * nDim,          cudaMemcpyHostToDevice );

    /* Allocate additional memory at the GPU */
	cudaMalloc( &d_nFacesPerPoint, sizeof(int)    * nPoints);
	cudaMalloc( &d_w,              sizeof(double) * nPoints * nDim );
	cudaMalloc( &d_logSqrt,        sizeof(double) * nPoints);
    cudaMalloc ((void**)&d_res, sizeof(double) * nPoints * nDim * nDim); 
    cudaMalloc( &d_gra1, sizeof(double) * nPoints * nDim );
    cudaMalloc( &d_gra2, sizeof(double) * nPoints * nDim );
    if (nDim == 3) cudaMalloc( &d_gra3, sizeof(double) * nPoints * nDim );
    cudaMalloc ((void**)&d_cg_tensor1, sizeof(double) * nPoints * nDim); 
    cudaMalloc ((void**)&d_cg_tensor2, sizeof(double) * nPoints * nDim); 
    if (nDim == 3) cudaMalloc ((void**)&d_cg_tensor3, sizeof(double) * nPoints * nDim); 
    cudaMalloc ( &d_ftl_matrix, sizeof(double)*nPoints * nDim * nDim );
    cudaMalloc ((void**)&d_A_ei, sizeof(double) * lda * m);            
    cudaMalloc ((void**)&d_W_ei, sizeof(double) * m * nPoints);            
    cudaMalloc ((void**)&devInfo, sizeof(int));

	/* Copy data to GPU */
	cudaMalloc( &d_facesPerPoint, sizeof(int) * nFacesPerPoint[ nPoints - 1 ]);
	cudaMemcpy( d_nFacesPerPoint, nFacesPerPoint, sizeof(int) * nPoints,                       cudaMemcpyHostToDevice );
	cudaMemcpy( d_facesPerPoint,  facesPerPoint,  sizeof(int) * nFacesPerPoint[ nPoints - 1 ], cudaMemcpyHostToDevice );
	
    //printf("Punto %d    caras en el punto %d \n", 0, nFacesPerPoint[0]);
    int maxi = nFacesPerPoint[0];
    for (int i=1; i<nPoints; i++)
    {
        if ((nFacesPerPoint[i]-nFacesPerPoint[i-1])>maxi ) maxi = nFacesPerPoint[i]-nFacesPerPoint[i-1];
        //printf("Punto %d    caras en el punto %d \n", i, nFacesPerPoint[i]-nFacesPerPoint[i-1]);
    }
    //printf("maxi %d\n",maxi);

	/* Create dim3 for GPU */
	dim3 block(blockSize);
    int numBlocks = (int) (ceil(    (double)nPoints/(double)block.x)  +1);
    numBlocks = numBlocks/omp_get_num_threads() + 1; 
	dim3 grid_numCoords(numBlocks);

    
    struct timeval start;
    struct timeval end;                 
    double time;
    int size =   omp_get_num_threads()==1 ?   0: nPoints / omp_get_num_threads() ;// nPoints / omp_get_num_threads();
    int localStride = size * omp_get_thread_num();
    int nDevices;
    cudaGetDeviceCount(&nDevices); 
    printf("Number of devices: %d   Number of blocks %d  size %d  localStrade %d\n", nDevices, grid_numCoords.x, size, localStride);
   
    /* Solve FTLE */
      cudaDeviceSynchronize();
      gettimeofday(&start, NULL);
   

//-----------------------------------------------------------------------------------------------------------------------------
    for(int i_repes=0; i_repes  < numRepes; i_repes++) {   
    /* STEP 1: compute gradient, tensors and ATxA based on neighbors flowmap values */
	if ( nDim == 2 )
        gpu_compute_gradient_2D <<<grid_numCoords, block>>> (localStride,
                                                              nPoints, nVertsPerFace, 
                                                              d_coords, d_flowmap, d_faces, d_nFacesPerPoint, d_facesPerPoint, 
                                                              d_ftl_matrix, d_W_ei, d_logSqrt, t_eval);
    else
        gpu_compute_gradient_3D <<<grid_numCoords, block>>> (localStride,
                                                              nPoints, nVertsPerFace, 
                                                              d_coords, d_flowmap, d_faces, d_nFacesPerPoint, d_facesPerPoint, 
                                                              d_ftl_matrix, d_W_ei, d_logSqrt, t_eval);

    /* Step 3: ftle <- log(srqt(max(eigen))) */
    /*if ( nDim == 2) 
        gpu_log_sqrt_2D <<<grid_numCoords, block>>> ( nPoints, t_eval, d_W_ei, d_logSqrt );
    else            
        gpu_log_sqrt_3D <<<grid_numCoords, block>>> ( nPoints, t_eval, d_W_ei, d_logSqrt );
    */
    
    cudaMemcpy (&logSqrt[localStride],  &d_logSqrt[localStride], sizeof(double) * (nPoints / omp_get_num_threads()), cudaMemcpyDeviceToHost);}
    
//-----------------------------------------------------------------------------------------------------------------------------  
//cudaDeviceSynchronize(); 
       /* Time */
       gettimeofday(&end, NULL);
       /* Show time */   
       time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1000000.0;
       //printf("Execution time..: %f\n", time);
   
        times[omp_get_thread_num()] = time;

        printf("Tiempo local:  %f  HILO %d, Numero de Bloques %d    localStride %d  Repeticiones %d\n", time, omp_get_thread_num(), numBlocks, localStride, numRepes);
        error = cudaGetLastError();
        if ( error != cudaSuccess )
                printf("ErrCUDA A: %s\n", cudaGetErrorString( error ) );
    /* Free memory */
    /*free(logSqrt);
    free(ftl_matrix);
    cudaFree(d_coords);
    cudaFree(d_flowmap);
    cudaFree(d_faces);
    cudaFree(d_nFacesPerPoint);
    cudaFree(d_facesPerPoint);
    cudaFree(d_w);
    cudaFree(d_logSqrt);
    cudaFree(d_ftl_matrix);
    cudaFree(d_gra1);
    cudaFree(d_gra2);
    cudaFree(d_gra3);
    cudaFree(d_res);
    cudaFree(d_A_ei);
    cudaFree(d_W_ei);
    cudaFree(d_cg_tensor1);
    cudaFree(d_cg_tensor2);
    cudaFree(d_cg_tensor3);
    cudaFree(devInfo);
    cudaFree(d_work);
    free(w);
    
    cusolverDnDestroy(cusolverH);
    cudaDeviceReset();*/

    }

        
        /*for ( int ii = 0; ii < nPoints ; ii++ )
        {
            printf("%d %f\n", ii, logSqrt[ii]);
        }*/
   

    double maxTime = times[0];
    for ( int i = 1; i< numThreads; i++){
        if (times[i] > maxTime)
        maxTime = times[i];
    }
    printf("TOTAL :::::::::::  Execution time: %f\n", maxTime);
    /*free(coords);
    free(flowmap);
    free(faces);
    free(nFacesPerPoint);
    free(facesPerPoint);*/
   
    
    

    return 0;
}
