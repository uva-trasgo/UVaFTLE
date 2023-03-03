#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <assert.h>
#include <vector>
#include <iostream>
#include <CL/sycl.hpp>

#include "ftle.h"
#include "arithmetic.h"
#include "preprocess.h"

#define blockSize 512
#define D1_RANGE(size) range<1>{static_cast<size_t>(size)}
#define HIP_PLATFORM 0
#define CUDA_PLATFORM 1
#define OMP_PLATFORM 2
#define ALL_GPUS_PLATFORM 3

using namespace cl::sycl;

std::vector<queue> get_queues_from_platform(int plat, int nGpus){
	if(plat == OMP_PLATFORM)
	{
		std::vector<queue> queues(1);
		queues[0] = queue(cpu_selector{});
		return queues;
	}
	if(plat == ALL_GPUS_PLATFORM){
		auto devs = device::get_devices(info::device_type::gpu);
		std::vector<queue> queues(nGpus);
		if(devs.size() < nGpus){
			 printf("ERROR: Requested %d GPUs, but only %d GPU available in the system. Aborting program...\n",nGpus,(int) devs.size());
			 exit(1);
		}
		for (int d=0; d< nGpus; d++){
			printf("Dispositivo %d: %s\n", d, devs[d].get_info<info::device::name>().c_str());
			queues[d] = queue(devs[d]);
		}
		return queues;
	}
	
	auto platform = platform::get_platforms();
	std::string check = (!plat) ? "HIP" : "CUDA";
	for (int p=0; p < platform.size(); p++){
		if(!platform[p].get_info<info::platform::name>().compare(check)){
			auto devs= platform[p].get_devices();
			if(devs.size() < nGpus){
			 	printf("ERROR: Requested %d GPUs, but only %d GPU available in the system. Aborting program...\n",nGpus,(int) devs.size());
			 	exit(1);
			}
			std::vector<queue> queues(nGpus);
			for (int d=0; d< nGpus; d++){
				printf("Dispositivo %d: %s\n", d, devs[d].get_info<info::device::name>().c_str());
				queues[d] = queue(devs[d]);
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
		printf("USAGE: %s <nDim> <coords_file> <faces_file> <flowmap_file> <t_eval> <print2file>\n", argv[0]);
		printf("\texecutable:    compute_ftle\n");
		printf("\tnDim:    dimensions of the space (2D/3D)\n");
		printf("\tcoords_file:   file where mesh coordinates are stored.\n");
		printf("\tfaces_file:    file where mesh faces are stored.\n");
		printf("\tflowmap_file:  file where flowmap values are stored.\n");
		printf("\tt_eval:        time when compute ftle is desired.\n");
		//printf("\tnth:           number of OpenMP threads to use.\n");
		printf("\tprint to file? (0-NO, 1-YES)\n");
		printf("\tNumber of GPUs [optional, only used for HIP and CUDA backends. Default: 1]\n");
		return 1;
	}

	struct timeval start;
	struct timeval end;
	struct timeval preprocess;
	double time;
	double t_eval = atof(argv[5]);
	int check_EOF;
	char buffer[255];

	int nDim, nVertsPerFace, nPoints, nFaces;

	double *coords, *flowmap;
	int    *faces;
	int    *nFacesPerPoint;

	int    *facesPerPoint;
#ifdef GPU_ALL	
	int    *facesPerPoint_aux;
#endif	
	double *w;
	double *logSqrt;


	/* Initialize mesh original information */
	nDim = atoi(argv[1]);
	if ( nDim == 2 ) nVertsPerFace = 3;    // 2D: faces are triangles
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
	printf("\tReading mesh points coordinates...        "); 
	fflush(stdout);
	FILE *file = fopen( argv[2], "r" );
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
	printf("\tReading mesh faces vertices...            "); 
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
	printf("\tReading mesh flowmap (x, y[, z])...       "); 
	fflush(stdout);
	flowmap = (double*) malloc( sizeof(double) * nPoints * nDim ); 
	read_flowmap ( argv[4], nDim, nPoints, flowmap );
	printf("DONE\n\n"); 
	fflush(stdout);

	/* Allocate additional memory at the CPU */   
	logSqrt        = (double*) malloc( sizeof(double) * nPoints);   
	nFacesPerPoint = (int *) malloc( sizeof(int) * nPoints ); /* REMARK: nFacesPerPoint accumulates previous nFacesPerPoint */
     /* Assign faces to vertices and generate nFacesPerPoint and facesPerPoint GPU vectors */
     create_nFacesPerPoint_vector ( nDim, nPoints, nFaces, nVertsPerFace, faces, nFacesPerPoint );
	facesPerPoint = (int *) malloc( sizeof(int) * nFacesPerPoint[ nPoints - 1 ] );
#ifdef GPU_ALL	
	facesPerPoint_aux = (int *) malloc( sizeof(int) * nFacesPerPoint[ nPoints - 1 ] );
#endif	
	/* Solve FTLE */
	fflush(stdout);

	/*Generate SYCL queues*/
	queue preproc_queue(gpu_selector{});
	queue q_cpu(cpu_selector{});
#ifdef HIP_DEVICE
	int nGpus = (argc==8) ?  atoi(argv[7]) : 1;
	auto queues = get_queues_from_platform(HIP_PLATFORM, nGpus);
#elif defined CUDA_DEVICE
	int nGpus = (argc==8) ?  atoi(argv[7]) : 1;
	auto queues = get_queues_from_platform(CUDA_PLATFORM, nGpus);
#elif defined GPU_ALL
	int nGpus = (argc==8) ?  atoi(argv[7]) : 1;
	auto queues = get_queues_from_platform(ALL_GPUS_PLATFORM, nGpus);
#else
	int nGpus = 1;
	auto queues = get_queues_from_platform(OMP_PLATFORM, nGpus);
#endif
	
	printf("Preprocessing device: %s\n", preproc_queue.get_device().get_info<info::device::name>().c_str());  
	for(int d =0; d < nGpus; d++)
		printf("Kernel device %d: %s\n", d, queues[d].get_device().get_info<info::device::name>().c_str());  
	gettimeofday(&start, NULL);

	printf("\nComputing FTLE (SYCL)...                     ");
	{  
		/*Creating SYCL BUFFERS*/
		cl::sycl::buffer<double, 1> b_coords(coords, D1_RANGE(nPoints * nDim)); //check
		cl::sycl::buffer<int, 1> b_faces(faces, D1_RANGE(nFaces * nVertsPerFace));  //check
		cl::sycl::buffer<double, 1> b_flowmap(flowmap, D1_RANGE(nPoints*nDim));//check
		cl::sycl::buffer<int, 1> b_nFacesPerPoint(nFacesPerPoint, D1_RANGE(nPoints)); //check
		cl::sycl::buffer<int, 1> b_facesPerPoint(facesPerPoint, D1_RANGE(nFacesPerPoint[ nPoints - 1 ])); //check
#ifdef GPU_ALL	
		cl::sycl::buffer<int, 1> b_facesPerPoint_aux(facesPerPoint_aux, D1_RANGE(nFacesPerPoint[ nPoints - 1 ])); //check
#endif
		std::vector<int> v_points(nGpus);
		std::vector<int> offsets(nGpus);
		int gap= nPoints / nGpus;
		for(int d=0; d < nGpus; d++){
			v_points[d] = gap; 
			if(d == nGpus-1)
				v_points[d]+=nPoints % nGpus;
			offsets[d] = gap*d;
		}
        	/*First Kernel for preprocessing */
#ifdef GPU_ALL	
		create_facesPerPoint_vector(&preproc_queue, nDim, nPoints, nFaces, nVertsPerFace, &b_faces, &b_nFacesPerPoint, &b_facesPerPoint_aux);
		q_cpu.submit([&](handler &h){
			auto faces1 = b_facesPerPoint_aux.get_access<access::mode::read>(h);
			auto faces2 = b_facesPerPoint.get_access<access::mode::discard_write>(h);
			h.parallel_for<class copy> (D1_RANGE(nFacesPerPoint[ nPoints - 1 ]), [=](id<1> i){
				faces2[i]=faces1[i];
			});
		});
		q_cpu.wait();
#else       	
		create_facesPerPoint_vector(&preproc_queue, nDim, nPoints, nFaces, nVertsPerFace, &b_faces, &b_nFacesPerPoint, &b_facesPerPoint );
		preproc_queue.wait();
#endif 		
		gettimeofday(&preprocess, NULL);
		
		
        /* Compute gradient, tensors and ATxA based on neighbors flowmap values, then get the max eigenvalue */
        for(int d=0; d < nGpus; d++){
			if ( nDim == 2 ){
			    cl::sycl::buffer<double, 1> b_logSqrt(logSqrt + offsets[d], D1_RANGE(v_points[d]));
		   	    compute_gradient_2D ( &queues[d], v_points[d], offsets[d], nVertsPerFace, &b_coords, &b_flowmap, &b_faces, &b_nFacesPerPoint, &b_facesPerPoint, &b_logSqrt, t_eval);
		  	}
			else{
			    cl::sycl::buffer<double, 1> b_logSqrt(logSqrt + offsets[d], D1_RANGE(v_points[d]));
		   	    compute_gradient_3D  ( &queues[d], v_points[d], offsets[d],  nVertsPerFace, &b_coords, &b_flowmap, &b_faces, &b_nFacesPerPoint, &b_facesPerPoint, &b_logSqrt, t_eval);
		   	}
	    }
	    for(int d =0; d < nGpus; d++)
	       	queues[d].wait();	
	}
	/* Time */
	gettimeofday(&end, NULL);
	printf("DONE\n\n");
	printf("--------------------------------------------------------\n");
	fflush(stdout);

	/* Print numerical results */
	if ( atoi(argv[6]) )
	{
		printf("\nWriting result in output file...                  ");
		fflush(stdout);
		FILE *fp_w = fopen("ftle_result.csv", "w");
		for ( int ii = 0; ii < nPoints; ii++ )
		{
			fprintf(fp_w, "%f\n", logSqrt[ii]);
		}
		fclose(fp_w);
		printf("DONE\n\n");
		printf("--------------------------------------------------------\n");
		fflush(stdout);
	}

	/* Show execution time */   
	/*TODO discutir que hacer con el tiempo de preprocesado*/
	time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1000000.0;
	printf("\nExecution time (seconds): %f\n",  time);
	time = (preprocess.tv_sec - start.tv_sec) + (preprocess.tv_usec - start.tv_usec)/1000000.0;
	printf("Preprocessing time (seconds): %f\n", time);
	time = (end.tv_sec - preprocess.tv_sec) + (end.tv_usec - preprocess.tv_usec)/1000000.0;
	printf("Processing time (seconds): %f\n\n",  time);
	printf("--------------------------------------------------------\n");
	fflush(stdout);

	/* Free memory */
	free(coords);
	free(flowmap);
	free(faces);
	free(nFacesPerPoint);
	free(facesPerPoint);
	free(logSqrt);
	return 0;
}
