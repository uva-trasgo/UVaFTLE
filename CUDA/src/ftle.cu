/*
 *            UVaFTLE 1.0: Lagrangian finite time 
 *		    Lyapunov exponent extraction 
 *		    for fluid dynamic applications
 *
 *    Copyright (C) 2023, 2024 Rocío Carratalá-Sáez et. al.
 *    This file is part of the UVaFTLE application.
 *
 *  UVaFTLE is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  UVaFTLE is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with UVaFTLE.  If not, see <http://www.gnu.org/licenses/>.
 */ 
 
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>

#include "ftle.h"
#include "arithmetic.h"
#include "preprocess.h"

#define BLOCK 512
#define Num_Streams 1


void show_GPU_devices_info(int nDevices)
{
	int i;
	cudaDeviceProp prop;
	for (i = 0; i < nDevices; i++) {
		cudaGetDeviceProperties(&prop, i);
		printf("GPU #%d\n", i);
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
	}
	fflush(stdout);
}

int main(int argc, char *argv[]) {

	printf("--------------------------------------------------------\n");
	printf("|                        UVaFTLE                       |\n");
	printf("|                                                      |\n");
	printf("| Developers:                                          |\n");
	printf("|  - Rocío Carratalá-Sáez | rocio@infor.uva.es         |\n");
	printf("|  - Yuri Torres          | yuri.torres@infor.uva.es   |\n");
	printf("|  - Sergio López-Huguet  | serlohu@upv.es             |\n");
	printf("|  - Francisco J. Andújar | fandujarm@infor.uva.es     |\n");
	printf("--------------------------------------------------------\n");
	fflush(stdout);

	// Check usage
	if (argc != 8)
	{
		printf("USAGE: %s <nDim> <coords_file> <faces_file> <flowmap_file> <t_eval> <print2file> <nDevices>\n", argv[0]);
		printf("\tnDim:    dimensions of the space (2D/3D)\n");
		printf("\tcoords_file:   file where mesh coordinates are stored.\n");
		printf("\tfaces_file:    file where mesh faces are stored.\n");
		printf("\tflowmap_file:  file where flowmap values are stored.\n");
		printf("\tt_eval:        time when compute ftle is desired.\n");
		printf("\tprint to file? (0-NO, 1-YES)\n");
                printf("\tnDevices:       number of GPUs\n");
		return 1;
	}

	double t_eval = atof(argv[5]);
	int check_EOF;
	int nDevices = atoi(argv[7]), maxDevices;
	char buffer[255];
	int nDim, nVertsPerFace, nPoints, nFaces;
	FILE *file;
	int	 numThreads=0;
	double *coords;
	double *flowmap;
	int	*faces;
	double *logSqrt;
	int	*nFacesPerPoint;
	int	*facesPerPoint;

	/* Obtain and show GPU devices information */
	printf("\nGPU devices to be used:		 \n\n");	
	fflush(stdout);
	cudaGetDeviceCount(&maxDevices);
	show_GPU_devices_info(maxDevices);
	printf("--------------------------------------------------------\n");
	fflush(stdout);
	float  kernel_times[maxDevices];
	float  preproc_times[maxDevices];

	/* Initialize mesh original information */
	nDim = atoi(argv[1]);
	if ( nDim == 2 ) nVertsPerFace = 3; // 2D: faces are triangles
	else {
		if ( nDim == 3) nVertsPerFace = 4; // 3D: faces (volumes) are tetrahedrons
		else
		{
			printf("Wrong dimension provided (2 or 3 supported)\n");
			return 1;
		}
	}

	/* Read coordinates, faces and flowmap from Python-generated files and generate corresponding GPU vectors */
	/* Read coordinates information */
	printf("\nReading input data\n\n"); 
	fflush(stdout);
	printf("\tReading mesh points coordinates...		"); 
	fflush(stdout);
	file = fopen( argv[2], "r" );
	check_EOF = fscanf(file, "%s", buffer);
	if ( check_EOF == EOF )
	{
		fprintf( stderr, "Error: Unexpected EOF in read_coordinates\n" ); 
		fflush(stdout);
		exit(-1);
	}
	nPoints = atoi(buffer);
	fclose(file);
	coords = (double *) malloc ( sizeof(double) * nPoints * nDim );
	read_coordinates(argv[2], nDim, nPoints, coords); 
	printf("DONE\n"); 
	fflush(stdout);

	/* Read faces information */
	printf("\tReading mesh faces vertices...			"); 
	fflush(stdout);
	file = fopen( argv[3], "r" );
	check_EOF = fscanf(file, "%s", buffer);
	if ( check_EOF == EOF )
	{
		fprintf( stderr, "Error: Unexpected EOF in read_faces\n" ); 
		fflush(stdout);
		exit(-1);
	}
	nFaces = atoi(buffer);
	faces = (int *) malloc ( sizeof(int) * nFaces * nVertsPerFace );
	read_faces(argv[3], nDim, nVertsPerFace, nFaces, faces); 
	printf("DONE\n"); 
	fflush(stdout);

	/* Read flowmap information */
	printf("\tReading mesh flowmap (x, y[, z])...	   "); 
	fflush(stdout);
	flowmap = (double*) malloc( sizeof(double) * nPoints * nDim ); 
	read_flowmap ( argv[4], nDim, nPoints, flowmap );
	printf("DONE\n\n"); 
	printf("--------------------------------------------------------\n"); 
	fflush(stdout);

	/* Allocate additional memory at the CPU */
	nFacesPerPoint = (int *) malloc( sizeof(int) * nPoints ); /* REMARK: nFacesPerPoint accumulates previous nFacesPerPoint */
	// Assign faces to vertices and generate nFacesPerPoint and facesPerPoint GPU vectors  
	create_nFacesPerPoint_vector ( nDim, nPoints, nFaces, nVertsPerFace, faces, nFacesPerPoint );
#ifdef PINNED
	cudaHostAlloc( (void **) &logSqrt, sizeof(double) * nPoints, cudaHostAllocMapped);
	cudaHostAlloc( (void **) &facesPerPoint, sizeof(int) * nFacesPerPoint[ nPoints - 1 ], cudaHostAllocMapped);
#else
	logSqrt= (double*) malloc( sizeof(double) * nPoints);
	facesPerPoint = (int *) malloc( sizeof(int) * nFacesPerPoint[ nPoints - 1 ] );
#endif
	
	
	int v_points[maxDevices];
	int offsets[maxDevices];
	int v_points_faces[maxDevices];
	int offsets_faces[maxDevices];
	int gap= ((nPoints / nDevices)/BLOCK)*BLOCK;
	for(int d=0; d < nDevices; d++){
		v_points[d] = (d == nDevices-1) ? nPoints - gap*d : gap; 
		offsets[d] = gap*d;
	}
	for(int d=0; d < nDevices; d++){
		int inf = (d != 0) ? nFacesPerPoint[offsets[d]-1] : 0;
		int sup = (d != nDevices-1) ? nFacesPerPoint[offsets[d+1]-1] : nFacesPerPoint[nPoints-1];
		v_points_faces[d] =  sup - inf;
		offsets_faces[d] = (d != 0) ? nFacesPerPoint[offsets[d]-1]: 0;
	}
	

#ifndef PINNED	
	printf("\nComputing FTLE (CUDA non pinned)...");
#else
	printf("\nComputing FTLE (CUDA pinned)...");
#endif
	struct timeval global_timer_start;
	gettimeofday(&global_timer_start, NULL);
	#pragma omp parallel default(none)  shared(stdout, logSqrt, nDim, nPoints, nFaces, nVertsPerFace, numThreads, preproc_times, kernel_times, v_points, v_points_faces, offsets, offsets_faces, faces, coords, nFacesPerPoint, facesPerPoint, flowmap, t_eval)  
	{
		numThreads = omp_get_num_threads();
		int d = omp_get_thread_num();
		double *d_logSqrt;
		double *d_coords, *d_flowmap;
		int *d_faces, *d_nFacesPerPoint, *d_facesPerPoint;

		cudaSetDevice(d);

		/* Allocate memory for read data at the GPU (coords, faces, flowmap) */
		cudaMalloc( &d_coords,  sizeof(double) * nPoints * nDim );
		cudaMalloc( &d_faces,   sizeof(int)	* nFaces  * nVertsPerFace );
		cudaMalloc( &d_flowmap, sizeof(double) * nPoints * nDim );

		/* Copy read data to GPU (coords, faces, flowmap) */
		cudaMemcpy( d_coords,  coords,  sizeof(double) * nPoints * nDim,		cudaMemcpyHostToDevice );
		cudaMemcpy( d_faces,   faces,   sizeof(int)	* nFaces  * nVertsPerFace,	cudaMemcpyHostToDevice );
		cudaMemcpy( d_flowmap, flowmap, sizeof(double) * nPoints * nDim,		cudaMemcpyHostToDevice );

		/* Allocate additional memory at the GPU */
		cudaMalloc( &d_nFacesPerPoint, sizeof(int)	* nPoints);
		cudaMalloc( &d_logSqrt,		sizeof(double) * v_points[d]); 

		/* Copy data to GPU */
		cudaMemcpy( d_nFacesPerPoint, nFacesPerPoint, sizeof(int) * nPoints,	cudaMemcpyHostToDevice );
		cudaMalloc( &d_facesPerPoint, sizeof(int) * v_points_faces[d]);		

		/* Create dim3 for GPU */
		dim3 block(BLOCK);
		int numBlocks = (int) (ceil((double)v_points[d]/(double)block.x)+1);
		dim3 grid_numCoords(numBlocks+1);

		//Create Cuda events
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);  	
		//Launch preproccesing kernel
		cudaEventRecord(start, cudaStreamDefault);
		/* STEP 1: compute gradient, tensors and ATxA based on neighbors flowmap values */
		create_facesPerPoint_vector<<<grid_numCoords, block, 0, cudaStreamDefault>>> (nDim, v_points[d], offsets[d], offsets_faces[d], nFaces, nVertsPerFace, d_faces, d_nFacesPerPoint, d_facesPerPoint);

		cudaEventRecord(stop, cudaStreamDefault);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(preproc_times+ omp_get_thread_num() , start, stop);
		cudaEventRecord(start, cudaStreamDefault);
		if ( nDim == 2 )
			gpu_compute_gradient_2D <<<grid_numCoords, block, 0, cudaStreamDefault>>> (v_points[d], offsets[d], offsets_faces[d], nVertsPerFace, d_coords, d_flowmap, d_faces, d_nFacesPerPoint, d_facesPerPoint, d_logSqrt, t_eval);
		else
			gpu_compute_gradient_3D <<<grid_numCoords, block, 0, cudaStreamDefault>>> (v_points[d], offsets[d], offsets_faces[d], nVertsPerFace, d_coords, d_flowmap, d_faces, d_nFacesPerPoint, d_facesPerPoint, d_logSqrt, t_eval);

		cudaEventRecord(stop, cudaStreamDefault);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(kernel_times + omp_get_thread_num(), start, stop);
	
#ifndef PINNED
 		cudaMemcpy (logSqrt +offsets[d],  d_logSqrt, sizeof(double) * v_points[d], cudaMemcpyDeviceToHost);
 		cudaMemcpy(facesPerPoint + offsets_faces[d], d_facesPerPoint,  sizeof(int) * v_points_faces[d], cudaMemcpyDeviceToHost );	
#else
		cudaMemcpyAsync (logSqrt + offsets[d],  d_logSqrt, sizeof(double) * v_points[d], cudaMemcpyDeviceToHost, cudaStreamDefault);
		cudaMemcpyAsync (facesPerPoint + offsets_faces[d], d_facesPerPoint,  sizeof(int) * v_points_faces[d], cudaMemcpyDeviceToHost, cudaStreamDefault);
		cudaDeviceSynchronize();
#endif
		
		fflush(stdout);
		
		/* Free memory */
		cudaFree(d_coords);
		cudaFree(d_flowmap);
		cudaFree(d_faces);
		cudaFree(d_nFacesPerPoint);
		cudaFree(d_facesPerPoint);
		cudaFree(d_logSqrt);
	}
	struct timeval global_timer_end;
	gettimeofday(&global_timer_end, NULL);
	double time = (global_timer_end.tv_sec - global_timer_start.tv_sec) + (global_timer_end.tv_usec - global_timer_start.tv_usec)/1000000.0;
	printf("DONE\n\n");
	printf("--------------------------------------------------------\n");
	fflush(stdout);
	/* Write result in output file (if desired) */
	if ( atoi(argv[6]) )
	{
		printf("\nWriting result in output file...				  ");
		fflush(stdout);
		FILE *fp_w = fopen("cuda_result.csv", "w");
		for ( int ii = 0; ii < nPoints; ii++ )
			fprintf(fp_w, "%f\n", logSqrt[ii]);
		fclose(fp_w);
		fp_w = fopen("cuda_preproc.csv", "w");
                for ( int ii = 0; ii < nFacesPerPoint[nPoints-1]; ii++ )
                        fprintf(fp_w, "%d\n", facesPerPoint[ii]);
                fclose(fp_w);
		printf("DONE\n\n");
		printf("--------------------------------------------------------\n");
		fflush(stdout);
	}

	/* Show execution time */
	printf("Execution times in miliseconds\n");
	printf("Device Num;  Preproc kernel; FTLE kernel\n");
	for(int d = 0; d < nDevices; d++){
		printf("%d; %f; %f\n", d, preproc_times[d], kernel_times[d]);
	}
	printf("Global time: %f:\n", time);
	printf("--------------------------------------------------------\n");
	fflush(stdout);
	
	/* Free memory */
	free(coords);
	free(faces);
	free(flowmap);
#ifndef PINNED			
	free(logSqrt);
	free(facesPerPoint);
#else
	cudaFree(logSqrt);
	cudaFree(facesPerPoint);
#endif
	free(nFacesPerPoint);

	return 0;
}
