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

#include <vector>
#include <iostream>
#include <CL/sycl.hpp>
#include <assert.h>

#include "ftle.h"
#include "arithmetic.h"
#include "preprocess.h"

#define maxDevices 4
#define D1_RANGE(size) range<1>{static_cast<size_t>(size)}
#define HIP_PLATFORM 0
#define CUDA_PLATFORM 1
#define OMP_PLATFORM 2
#define ALL_GPUS_PLATFORM 3

using namespace cl::sycl;

float getKernelExecutionTime(::event event){
	auto start_time = event.get_profiling_info<::info::event_profiling::command_start>();
 	auto end_time = event.get_profiling_info<cl::sycl::info::event_profiling::command_end>();
 	return (end_time - start_time) / 1000000.0f;
}

std::vector<queue> get_queues_from_platform(int plat, int nDevices, int device_order){
	auto property_list =::property_list{::property::queue::enable_profiling()};
	if(plat == OMP_PLATFORM)
	{
		std::vector<queue> queues(1);
		queues[0] = queue(cpu_selector{}, property_list);
		return queues;
	}
	if(plat == ALL_GPUS_PLATFORM){
		auto devs = device::get_devices(info::device_type::gpu);
		std::vector<queue> queues(nDevices);
		if(devs.size() < nDevices){
			 printf("ERROR: Requested %d GPUs, but only %d GPU available in the system. Aborting program...\n",nDevices,(int) devs.size());
			 exit(1);
		}
		for (int d=0; d< nDevices; d++){
			int dd = (device_order) ? d :  devs.size() - 1 -d;
			printf("Dispositivo %d: %s\n", d, devs[dd].get_info<info::device::name>().c_str());
			queues[d] = queue(devs[dd], property_list);
		}
		return queues;
	}
	
	auto platform = platform::get_platforms();
	std::string check = (!plat) ? "HIP" : "CUDA";
	for (int p=0; p < platform.size(); p++){
		if(!platform[p].get_info<info::platform::name>().compare(check)){
			auto devs= platform[p].get_devices();
			if(devs.size() < nDevices){
			 	printf("ERROR: Requested %d GPUs, but only %d GPU available in the system. Aborting program...\n",nDevices,(int) devs.size());
			 	exit(1);
			}
			std::vector<queue> queues(nDevices);
			for (int d=0; d< nDevices; d++){
				printf("Dispositivo %d: %s\n", d, devs[d].get_info<info::device::name>().c_str());
				queues[d] = queue(devs[d], property_list);
			}
			return queues;
		}
	}
	return std::vector<queue>();
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
	if (argc < 8)
	{
		printf("USAGE: %s <nDim> <coords_file> <faces_file> <flowmap_file> <t_eval> <print2file> <nDevices>\n", argv[0]);
		printf("\tnDim:    dimensions of the space (2D/3D)\n");
		printf("\tcoords_file:   file where mesh coordinates are stored.\n");
		printf("\tfaces_file:    file where mesh faces are stored.\n");
		printf("\tflowmap_file:  file where flowmap values are stored.\n");
		printf("\tt_eval:        time when compute ftle is desired.\n");
		printf("\tprint to file? (0-NO, 1-YES)\n");
          	printf("\tnDevices:       number of GPUs\n");
#ifdef GPU_ALL
		printf("\tDevice order:    (0 - from 0 to n-1; from n-1 to 0\n");    
#endif		      
		return 1;
	}

	double t_eval = atof(argv[5]);
	int check_EOF;
	int nDevices = atoi(argv[7]);
	int device_order = (argc == 9) ? atoi(argv[8]): 0;
	char buffer[255];
	int nDim, nVertsPerFace, nPoints, nFaces;
	FILE *file;
	double *coords;
	double *flowmap;
	int	*faces;
	double *logSqrt[nDevices];	
	int	*nFacesPerPoint;
	int	*facesPerPoint[nDevices];

	/*Generate SYCL queues*/
#ifdef 	HIP_DEVICE
	auto queues = get_queues_from_platform(HIP_PLATFORM, nDevices, 0);
#elif 	defined CUDA_DEVICE
	auto queues = get_queues_from_platform(CUDA_PLATFORM, nDevices, 0);
#elif 	defined GPU_ALL
	auto queues = get_queues_from_platform(ALL_GPUS_PLATFORM, nDevices,device_order);
#else
	nDevices = 1;
	auto queues = get_queues_from_platform(OMP_PLATFORM, nDevices,0);
#endif
	for(int d =0; d < nDevices; d++)
		printf("Kernel device %d: %s\n", d, queues[d].get_device().get_info<info::device::name>().c_str());  

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
	coords = malloc_shared<double> (nPoints * nDim, queues[0]);
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
	faces = malloc_shared<int> (nFaces * nVertsPerFace, queues[0]);
	read_faces(argv[3], nDim, nVertsPerFace, nFaces, faces); 
	printf("DONE\n"); 
	fflush(stdout);

	/* Read flowmap information */
	printf("\tReading mesh flowmap (x, y[, z])...	   "); 
	fflush(stdout);
	flowmap = malloc_shared<double>(nPoints * nDim, queues[0]); 
	read_flowmap ( argv[4], nDim, nPoints, flowmap );
	printf("DONE\n\n"); 
	printf("--------------------------------------------------------\n"); 
	fflush(stdout);

	/* Allocate additional memory at the CPU */
	nFacesPerPoint = malloc_shared<int>(nPoints, queues[0]); /* REMARK: nFacesPerPoint accumulates previous nFacesPerPoint */
	// Assign faces to vertices and generate nFacesPerPoint and facesPerPoint GPU vectors  
	create_nFacesPerPoint_vector ( nDim, nPoints, nFaces, nVertsPerFace, faces, nFacesPerPoint );
	int v_points[nDevices];
	int offsets[nDevices];
	int v_points_faces[nDevices];
	int offsets_faces[nDevices];
	::event event_list[nDevices*2];
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
	for(int d=0; d < nDevices; d++){
		logSqrt[d]= malloc_shared<double>(v_points[d], queues[d]);
		facesPerPoint[d] = malloc_shared<int>(v_points_faces[d], queues[d]);
	}
	
	printf("\nComputing FTLE (SYCL USM SPLIT)...");
	struct timeval global_timer_start;
	gettimeofday(&global_timer_start, NULL);

	{
		
		/* STEP 1: compute gradient, tensors and ATxA based on neighbors flowmap values */
		for(int d=0; d < nDevices; d++){
			event_list[d] = create_facesPerPoint_vector(&queues[d], nDim, v_points[d], offsets[d], offsets_faces[d], nFaces, nVertsPerFace, faces, nFacesPerPoint, facesPerPoint[d]);

			if ( nDim == 2 )
				event_list[nDevices + d] = compute_gradient_2D ( &event_list[d], &queues[d], v_points[d], offsets[d], offsets_faces[d], nVertsPerFace, coords, flowmap, faces, nFacesPerPoint,facesPerPoint[d],logSqrt[d], t_eval);
		  	else
				event_list[nDevices + d] = compute_gradient_3D  ( &event_list[d], &queues[d], v_points[d], offsets[d], offsets_faces[d], nVertsPerFace, coords, flowmap, faces, nFacesPerPoint,facesPerPoint[d], logSqrt[d], t_eval);
		   	
		}
		for(int d=0; d < nDevices; d++)
			event_list[nDevices + d].wait();
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
		FILE *fp_w = fopen("usm_split_result.csv", "w");
		for(int d=0; d < nDevices; d++)
			for ( int ii = 0; ii < v_points[d]; ii++ )
				fprintf(fp_w, "%f\n", logSqrt[d][ii]);
		fclose(fp_w);
		fp_w = fopen("usm_split_preproc.csv", "w");
		for(int d=0; d < nDevices; d++)
                for ( int ii = 0; ii < v_points_faces[d]; ii++ )
                        fprintf(fp_w, "%d\n", facesPerPoint[d][ii]);
          fclose(fp_w);
		printf("DONE\n\n");
		printf("--------------------------------------------------------\n");
		fflush(stdout);
	}

	/* Show execution time */
	printf("Execution times in miliseconds\n");
	printf("Device Num;  Preproc kernel; FTLE kernel\n");
	for(int d = 0; d < nDevices; d++){
		printf("%d; %f; %f\n", d, getKernelExecutionTime(event_list[d]), getKernelExecutionTime(event_list[nDevices + d]));
	}
	printf("Global time: %f:\n", time);
	printf("--------------------------------------------------------\n");
	fflush(stdout);
	
	/* Free memory */
	free(coords, queues[0]);
	free(faces, queues[0]);
	free(flowmap, queues[0]);
	free(nFacesPerPoint, queues[0]);
	for(int d=0; d < nDevices; d++){
		free(logSqrt[d], queues[d]);
		free(facesPerPoint[d], queues[d]);
	}

	return 0;
}
