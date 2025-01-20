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
#include <sycl/sycl.hpp>


#include "ftle.h"
#include "arithmetic.h"
#include "preprocess.h"

#define maxDevices 4
#define D1_RANGE(size) range<1>{static_cast<size_t>(size)}
#define HIP_PLATFORM 0
#define CUDA_PLATFORM 1
#define OMP_PLATFORM 2
#define ALL_GPUS_PLATFORM 3

using namespace sycl;

float getKernelExecutionTime(event event){
	auto start_time = event.get_profiling_info<info::event_profiling::command_start>();
 	auto end_time = event.get_profiling_info<info::event_profiling::command_end>();
 	return (end_time - start_time) / 1000000.0f;
}

std::vector<queue> get_queues_from_platform(int plat, int nDevices, int device_order){
	auto my_property_list =property_list{ property::queue::enable_profiling(), property::queue::in_order()};
	if(plat == OMP_PLATFORM)
	{
		std::vector<queue> queues(1);
		queues[0] = queue(cpu_selector_v, my_property_list);
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
			queues[d] = queue(devs[dd], my_property_list);
		}
		return queues;
	}
#ifdef ONEAPI
	auto platform = platform::get_platforms();
	std::string check = (!plat) ? "HIP" : "NVIDIA CUDA BACKEND";
	int num_dev_found=0;
	std::vector<queue> queues(nDevices);
	for (int p=0; p < platform.size() && num_dev_found < nDevices; p++){
		printf("Plataforma %d: %s\n", p, platform[p].get_info<info::platform::name>().c_str());
		if(!platform[p].get_info<info::platform::name>().compare(check)){
			auto devs= platform[p].get_devices();	
			for (int d=0; d< devs.size() && num_dev_found < nDevices; d++){
				printf("Dispositivo %d: %s\n", d, devs[d].get_info<info::device::name>().c_str());
				queues[num_dev_found] = queue(devs[d], my_property_list);
				num_dev_found++;
			}
		}	
	}
	if(num_dev_found < nDevices){
		printf("ERROR: Requested %d GPUs, but only %d GPU available in the system. Aborting program...\n",nDevices,num_dev_found);
		exit(1);
	}
	
	return queues;
#else
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
				queues[d] = queue(devs[d], my_property_list);
			}
			return queues;
		}
	}
	return std::vector<queue>();
#endif
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
	double *logSqrt;
	int	*nFacesPerPoint;
	int	*facesPerPoint;

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
	logSqrt= (double*) malloc( sizeof(double) * nPoints);
	facesPerPoint = (int *) malloc( sizeof(int) * nFacesPerPoint[ nPoints - 1 ] );
	int v_points[nDevices];
	int offsets[nDevices];
	int v_points_faces[nDevices];
	int offsets_faces[nDevices];
	event event_list[nDevices*2];
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
	printf("\nComputing FTLE (SYCL USM DEVICE)...");
	struct timeval global_timer_start;
	gettimeofday(&global_timer_start, NULL);

	{
		
		double* d_coords[nDevices];
		int * d_faces[nDevices];
		double * d_flowmap[nDevices];
		/* Allocate additional memory at the GPU */
		int * d_nFacesPerPoint[nDevices];
		int * d_facesPerPoint[nDevices];
		double * d_logSqrt[nDevices];

		/* STEP 1: compute gradient, tensors and ATxA based on neighbors flowmap values */
		for(int d=0; d < nDevices; d++){
			d_coords[d] = malloc_device<double> (nPoints * nDim, queues[d]);
			d_faces[d] = malloc_device<int> (nFaces  * nVertsPerFace, queues[d]);
			d_flowmap[d] = malloc_device<double> (nPoints * nDim, queues[d]);
			/* Allocate additional memory at the GPU */
			d_nFacesPerPoint[d] = malloc_device<int> (nPoints, queues[d]);
			d_facesPerPoint[d] = malloc_device<int> (v_points_faces[d], queues[d]);
			d_logSqrt[d] = malloc_device<double> (v_points[d], queues[d]);
			queues[d].memcpy( d_coords[d],  coords,  sizeof(double) * nPoints * nDim);
			queues[d].memcpy( d_faces[d],   faces,   sizeof(int)	* nFaces  * nVertsPerFace);
			queues[d].memcpy( d_flowmap[d], flowmap, sizeof(double) * nPoints * nDim);
			queues[d].memcpy( d_nFacesPerPoint[d], nFacesPerPoint, sizeof(int) * nPoints);

			event_list[d] = create_facesPerPoint_vector(&queues[d], nDim, v_points[d], offsets[d], offsets_faces[d], nFaces, nVertsPerFace, d_faces[d], d_nFacesPerPoint[d], d_facesPerPoint[d]);
		
		}
		for(int d=0; d < nDevices; d++){	
			if ( nDim == 2 )
				event_list[nDevices + d] = compute_gradient_2D (&queues[d], v_points[d], offsets[d], offsets_faces[d], nVertsPerFace, d_coords[d], d_flowmap[d], d_faces[d], d_nFacesPerPoint[d],d_facesPerPoint[d],d_logSqrt[d], t_eval);
		  	else
				event_list[nDevices + d] = compute_gradient_3D  (&queues[d], v_points[d], offsets[d], offsets_faces[d], nVertsPerFace, d_coords[d], d_flowmap[d], d_faces[d], d_nFacesPerPoint[d],d_facesPerPoint[d],d_logSqrt[d], t_eval);
		}
		for(int d=0; d < nDevices; d++){	 	
			queues[d].memcpy(logSqrt +offsets[d],  d_logSqrt[d], sizeof(double) * v_points[d]);
			queues[d].memcpy(facesPerPoint + offsets_faces[d],  d_facesPerPoint[d], sizeof(int) * v_points_faces[d]);
 			
 		}
		
		for(int d=0; d < nDevices; d++){
 			queues[d].wait();
 			 
 			free(d_coords[d], queues[d]);
			free(d_flowmap[d], queues[d]);
			free(d_faces[d], queues[d]);
			free(d_nFacesPerPoint[d], queues[d]);
			free(d_facesPerPoint[d], queues[d]);
			free(d_logSqrt[d], queues[d]);
		}
		
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
#ifdef ONEAPI
		FILE *fp_w = fopen("OP_usm_result.csv", "w");
#else
		FILE *fp_w = fopen("usm_result.csv", "w");
#endif
		
		for ( int ii = 0; ii < nPoints; ii++ )
			fprintf(fp_w, "%f\n", logSqrt[ii]);
		fclose(fp_w);
#ifdef ONEAPI
		fp_w = fopen("OP_usm_preproc.csv", "w");
#else
		fp_w = fopen("usm_preproc.csv", "w");
#endif
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
		printf("%d; %f; %f\n", d, getKernelExecutionTime(event_list[d]), getKernelExecutionTime(event_list[nDevices + d]));
	}
	printf("Global time: %f:\n", time);
	printf("--------------------------------------------------------\n");
	printf("Event Timestamp\n");
	std::cout << "Device Name; Preproc Start; Preproc End; FTLE Start; FTLE END" << std::endl;
	for(int d = 0; d < nDevices; d++){
		auto start_pre = event_list[d].get_profiling_info<info::event_profiling::command_start>();
 		auto end_pre = event_list[d].get_profiling_info<info::event_profiling::command_end>();
 		auto start_ftle = event_list[nDevices + d].get_profiling_info<info::event_profiling::command_start>();
 		auto end_ftle = event_list[nDevices + d].get_profiling_info<info::event_profiling::command_end>();
 	
 		std::cout << queues[d].get_device().get_info<info::device::name>().c_str() << ";" << std::fixed  <<  start_pre  << ";" << std::fixed  <<  end_pre << ";" << std::fixed  <<  start_ftle << ";" << std::fixed  <<  end_ftle << std::endl;
 	}
 	printf("--------------------------------------------------------\n");
	fflush(stdout);
	
	/* Free memory */
	free(coords);
	free(faces);
	free(flowmap);
	free(logSqrt);
	free(facesPerPoint);
	free(nFacesPerPoint);
	return 0;
}
