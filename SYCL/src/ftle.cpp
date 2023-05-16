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

std::vector<queue> get_queues_from_platform(int plat, int nDevices){
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
			printf("Dispositivo %d: %s\n", d, devs[nDevices - 1 -d ].get_info<info::device::name>().c_str());
			queues[d] = queue(devs[nDevices - 1 -d ], property_list);
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
	if (argc != 7 && argc != 8)
	{
		printf("USAGE: %s <nDim> <coords_file> <faces_file> <flowmap_file> <t_eval> <print2file> <nDevices>\n", argv[0]);
		printf("\tnDim:    dimensions of the space (2D/3D)\n");
		printf("\tcoords_file:   file where mesh coordinates are stored.\n");
		printf("\tfaces_file:    file where mesh faces are stored.\n");
		printf("\tflowmap_file:  file where flowmap values are stored.\n");
		printf("\tt_eval:        time when compute ftle is desired.\n");
		printf("\tprint to file? (0-NO, 1-YES)\n");
		printf("\tnDevices:       number of GPUs [optional, only used for HIP and CUDA backends. Default: 1]\n");
		return 1;
	}

	double t_eval = atof(argv[5]);
	int check_EOF;
	int nDevices = (argc==8) ?  atoi(argv[7]) : 1;
	char buffer[255];
	int nDim, nVertsPerFace, nPoints, nFaces;
	FILE *file;

	double *coords;
	double *flowmap;
	int	*faces;
	double *logSqrt;
	int	*nFacesPerPoint;
	int	*facesPerPoint;

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
	logSqrt= (double*) malloc( sizeof(double) * nPoints);
	// Assign faces to vertices and generate nFacesPerPoint and facesPerPoint GPU vectors  
	create_nFacesPerPoint_vector ( nDim, nPoints, nFaces, nVertsPerFace, faces, nFacesPerPoint );
	facesPerPoint = (int *) malloc( sizeof(int) * nFacesPerPoint[ nPoints - 1 ] );

	/*Generate SYCL queues*/
#ifdef 	HIP_DEVICE
	auto queues = get_queues_from_platform(HIP_PLATFORM, nDevices);
#elif 	defined CUDA_DEVICE
	auto queues = get_queues_from_platform(CUDA_PLATFORM, nDevices);
#elif 	defined GPU_ALL
	auto queues = get_queues_from_platform(ALL_GPUS_PLATFORM, nDevices);
#else
	nDevices = 1;
	auto queues = get_queues_from_platform(OMP_PLATFORM, nDevices);
#endif

	std::vector<int> v_points(maxDevices);
	std::vector<int> offsets(maxDevices);
	std::vector<int> v_points_faces(maxDevices);
	std::vector<int> offsets_faces(maxDevices);
	std::vector<::event> event_list(nDevices*2);
	int gap= ((nPoints / nDevices)/BLOCK)*BLOCK;
	for(int d=0; d < maxDevices; d++){
		if(d < nDevices){
			v_points[d] = (d == nDevices-1) ? nPoints - gap*d : gap; 
			offsets[d] = gap*d;
		}
		else{
			v_points[d] = 1; 
			offsets[d] = 0;
		}
	}
	for(int d=0; d < maxDevices; d++){
		if(d < nDevices){
			int inf = (d != 0) ? nFacesPerPoint[offsets[d]-1] : 0;
			int sup = (d != nDevices-1) ? nFacesPerPoint[offsets[d+1]-1] :nFacesPerPoint[nPoints-1];
			v_points_faces[d] =  sup - inf;
			offsets_faces[d] = (d != 0) ? nFacesPerPoint[offsets[d]-1]: 0;
		}
		else{
			v_points_faces[d] = 1; 
			offsets_faces[d] =  0;
		}
		printf("gpu %d,  offset %d, elements %d\n", d,offsets[d], v_points[d]);
		printf("gpu %d,  offset_faces %d, elements_faces %d\n", d,offsets_faces[d], v_points_faces[d]);
	}
	
	for(int d =0; d < nDevices; d++)
		printf("Kernel device %d: %s\n", d, queues[d].get_device().get_info<info::device::name>().c_str());  

	printf("\nComputing FTLE (SYCL)...");
	struct timeval global_timer_start;
	gettimeofday(&global_timer_start, NULL);

	{
		/*Creating SYCL BUFFERS*/
		::buffer<double, 1> b_coords(coords, D1_RANGE(nPoints * nDim)); 
		::buffer<int, 1> b_faces(faces, D1_RANGE(nFaces * nVertsPerFace)); 
		::buffer<double, 1> b_flowmap(flowmap, D1_RANGE(nPoints*nDim));
		::buffer<int, 1> b_nFacesPerPoint(nFacesPerPoint, D1_RANGE(nPoints)); 
		::buffer<int, 1> b_faces0(facesPerPoint + offsets_faces[0], D1_RANGE(v_points_faces[0]));	
		::buffer<int, 1> b_faces1(facesPerPoint + offsets_faces[1], D1_RANGE(v_points_faces[1]));	
		::buffer<int, 1> b_faces2(facesPerPoint + offsets_faces[2], D1_RANGE(v_points_faces[2]));
		::buffer<int, 1> b_faces3(facesPerPoint + offsets_faces[3], D1_RANGE(v_points_faces[3]));		
		::buffer<double, 1> b_logSqrt0(logSqrt + offsets[0], D1_RANGE(v_points[0]));
		::buffer<double, 1> b_logSqrt1(logSqrt + offsets[1], D1_RANGE(v_points[1]));		
		::buffer<double, 1> b_logSqrt2(logSqrt + offsets[2], D1_RANGE(v_points[2]));
		::buffer<double, 1> b_logSqrt3(logSqrt + offsets[3], D1_RANGE(v_points[3]));	
		
        	/*First Kernel for preprocessing */
		for(int d=0; d < nDevices; d++){
			event_list[d] = create_facesPerPoint_vector(&queues[d], nDim, v_points[d], offsets[d], offsets_faces[d], nFaces, nVertsPerFace, &b_faces, &b_nFacesPerPoint,
			(d==0 ? &b_faces0 : (d==1 ? &b_faces1 : (d==2 ? &b_faces2 : &b_faces3))));
		}
		
        /* Compute gradient, tensors and ATxA based on neighbors flowmap values, then get the max eigenvalue */
       	for(int d=0; d < nDevices; d++){
			if ( nDim == 2 ){
				event_list[nDevices + d] = compute_gradient_2D ( &queues[d], v_points[d], offsets[d], offsets_faces[d], nVertsPerFace, &b_coords, &b_flowmap, &b_faces, &b_nFacesPerPoint,(d==0 ? &b_faces0 : (d==1 ? &b_faces1 : (d==2 ? &b_faces2 : &b_faces3))),(d==0 ? &b_logSqrt0 : (d==1 ? &b_logSqrt1 : (d==2 ? &b_logSqrt2 : &b_logSqrt3))), t_eval);
		  	}else{
				event_list[nDevices + d] = compute_gradient_3D  ( &queues[d], v_points[d], offsets[d], offsets_faces[d], nVertsPerFace, &b_coords, &b_flowmap, &b_faces, &b_nFacesPerPoint,(d==0 ? &b_faces0 : (d==1 ? &b_faces1 : (d==2 ? &b_faces2 : &b_faces3))), (d==0 ? &b_logSqrt0 : (d==1 ? &b_logSqrt1 : (d==2 ? &b_logSqrt2 : &b_logSqrt3))), t_eval);
		   	}
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
		FILE *fp_w = fopen("sycl_result.csv", "w");
		for ( int ii = 0; ii < nPoints; ii++ )
			fprintf(fp_w, "%f\n", logSqrt[ii]);
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
	free(coords);
	free(faces);
	free(flowmap);
	free(logSqrt);
	free(nFacesPerPoint);
	free(facesPerPoint);

	return 0;
}
