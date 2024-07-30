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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#include "ftle.h"
#include "preprocess.h"

#include "opencl_helper.h"

int main(int argc, char *argv[])
{

	printf("--------------------------------------------------------\n");
	printf("|                        UVaFTLE                       |\n");
	printf("|                                                      |\n");
	printf("| Developers:                                          |\n");
	printf("|  - Rocío Carratalá-Sáez | rcarrata@uji.es            |\n");
	printf("|  - Yuri Torres          | yuri.torres@infor.uva.es   |\n");
	printf("|  - Sergio López-Huguet  | serlohu@upv.es             |\n");
	printf("|  - Manuel de Castro     | manuel@infor.uva.es        |\n");
	printf("--------------------------------------------------------\n");
	fflush(stdout);

	// Check usage
	if (argc != 7 || argc != 9)
	{
		printf("USAGE: %s <nDim> <coords_file> <faces_file> <flowmap_file> <t_eval> <print2file> [nFaces_file, faces_file]\n", argv[0]);
		printf("\texecutable:    compute_ftle.\n");
		printf("\tnDim:          dimensions of the space (2D/3D).\n");
		printf("\tcoords_file:   file where mesh coordinates are stored.\n");
		printf("\tfaces_file:    file where mesh faces are stored.\n");
		printf("\tflowmap_file:  file where flowmap values are stored.\n");
		printf("\tt_eval:        time when compute ftle is desired.\n");
		printf("\tprint2file:    print to file? (0-NO, 1-YES).\n");
		printf("\t[nFaces_file]: nFaces_per_point precomputed file.\n");
		printf("\t[faces_file]:  faces_per_point precomputed file.\n");
		exit(EXIT_FAILURE);
	}

	struct timeval preproc_clock, ftle_clock, end_clock;
	// Additional timers for FPGA performance analysis:
	struct timeval cpu2fpga_transfer_begin, kernel_begin, fpga2cpu_transfer_begin, fpga_end;
	double time_ftle, time_total;
	int check_EOF;
	char buffer[255];
	int nDim, nPoints, nFaces;
	int nVertsPerFace;
	FILE *file;

	double t_eval = atof(argv[5]);
	double *coords;
	double *flowmap;
	int *faces;
	double *logSqrt;

	int *nFacesPerPoint;
	int *facesPerPoint;

	/* Initialize mesh original information */
	nDim = atoi(argv[1]);
	if (nDim == 2)
		nVertsPerFace = 3; // 2D: faces are triangles
	else if (nDim == 3)
		nVertsPerFace = 4; // 3D: faces (volumes) are tetrahedron
	else
	{
		printf("Wrong dimension provided (2 or 3 supported)\n");
		exit(EXIT_FAILURE);
	}

	/* Read coordinates, faces and flowmap from Python-generated files and generate corresponding vectors */
	/* Read coordinates information */
	printf("\nReading input data\n\n");
	printf("\tReading mesh points coordinates...        ");
	fflush(stdout);

	file = fopen(argv[2], "r");
	check_EOF = fscanf(file, "%s", buffer);
	if (check_EOF == EOF)
	{
		fprintf(stderr, "Error: Unexpected EOF in read_coordinates\n");
		fflush(stdout);
		exit(EXIT_FAILURE);
	}
	nPoints = atoi(buffer);
	fclose(file);
	coords = (double *)malloc(sizeof(double) * nPoints * nDim);
	read_coordinates(argv[2], nDim, nPoints, coords);
	printf("DONE\n");
	fflush(stdout);

	/* Read faces information */
	printf("\tReading mesh faces vertices...            ");
	fflush(stdout);
	file = fopen(argv[3], "r");
	check_EOF = fscanf(file, "%s", buffer);
	if (check_EOF == EOF)
	{
		fprintf(stderr, "Error: Unexpected EOF in read_faces\n");
		fflush(stdout);
		exit(EXIT_FAILURE);
	}
	nFaces = atoi(buffer);
	faces = (int *)malloc(sizeof(int) * nFaces * nVertsPerFace);
	read_faces(argv[3], nDim, nVertsPerFace, nFaces, faces);
	printf("DONE\n");
	fflush(stdout);

	/* Read flowmap information */
	printf("\tReading mesh flowmap (x, y[, z])...       ");
	fflush(stdout);
	flowmap = (double *)malloc(sizeof(double) * nPoints * nDim);
	read_flowmap(argv[4], nDim, nPoints, flowmap);
	printf("DONE\n\n");
	printf("--------------------------------------------------------\n");
	fflush(stdout);

	/* Allocate additional memory at the CPU */
	logSqrt = (double *)malloc(sizeof(double) * nPoints);
	nFacesPerPoint = (int *)malloc(sizeof(int) * nPoints); /* REMARK: nFacesPerPoint accumulates previous nFacesPerPoint */

	/* Create OpenCL context and queue for FPGA and retrieve kernels */
	cl_int err;
	cl_device_id fpga_id;

	char kernel_directory[FILENAME_MAX];
	strncpy(kernel_directory, argv[0], FILENAME_MAX);
	char *kernel_directory_tail = kernel_directory + strlen(kernel_directory) - 1;
	while (&kernel_directory_tail[1] != kernel_directory && kernel_directory_tail[0] != '/')
		kernel_directory_tail--;
	kernel_directory_tail[1] = '\0';

	char *kernel_file_name = strcat(kernel_directory,
#ifndef EMULATION
	                                "arithmetic.aocx"
#else  // EMULATION
	                                "arithmetic_emu.aocx"
#endif // EMULATION
	);
	printf("Loading kernel file: %s...\n", kernel_file_name);
	fflush(stdout);

	cl_context ctx = cl_context_and_device_for_platform(
#ifdef EMULATION
	    "Intel(R) FPGA Emulation Platform for OpenCL(TM)",
#else  // !EMULATION
	    "Intel(R) FPGA SDK for OpenCL(TM)",
#endif // EMULATION
	    &fpga_id);

	cl_command_queue queue = clCreateCommandQueue(ctx, fpga_id, CL_QUEUE_PROFILING_ENABLE, &err);
	OPENCL_CHECK_ERROR(err);

	const char *kernel_names[2] = {"fpga_compute_gradient_2D", "fpga_compute_gradient_3D"};
	cl_kernel *fpga_kernels = cl_kernels_from_names(kernel_file_name, ctx, fpga_id, 2, kernel_names);
	cl_kernel fpga_compute_gradient_2D = fpga_kernels[0];
	cl_kernel fpga_compute_gradient_3D = fpga_kernels[1];
	cl_kernel fpga_compute_gradient; // Will store the kernel to execute
	free(fpga_kernels);

	/* PREPROCESS */
	/* Assign faces to vertices and generate nFacesPerPoint and facesPerPoint vectors */
	if (argc == 7)
	{
		create_nFacesPerPoint_vector(nDim, nPoints, nFaces, nVertsPerFace, faces, nFacesPerPoint);
		facesPerPoint = (int *)malloc(sizeof(int) * nFacesPerPoint[nPoints - 1]);
		gettimeofday(&preproc_clock, NULL);
		printf("\nComputing Preproc...                     ");
		create_facesPerPoint_vector(nDim, nPoints, nFaces, nVertsPerFace, faces, nFacesPerPoint, facesPerPoint);
	}
	else // (argc == 9)
	{
		read_nFacesPerPoint(argv[7], nDim, nPoints, nFacesPerPoint);
		facesPerPoint = (int *)malloc(sizeof(int) * nFacesPerPoint[nPoints - 1]);
		gettimeofday(&preproc_clock, NULL);
		printf("\nReading Preproc files...                 ");
		read_facesPerPoint(argv[8], nDim, nFacesPerPoint[nPoints - 1], facesPerPoint);
	}

	cl_event kernel_event;

	printf("\nComputing FTLE...                                 ");
	fflush(stdout);

	cl_mem d_logSqrt;
	cl_mem d_coords, d_flowmap;
	cl_mem d_faces, d_nFacesPerPoint, d_facesPerPoint;

	gettimeofday(&cpu2fpga_transfer_begin, NULL);
	/* Allocate and copy memory for read data at the FPGA (coords, faces, flowmap) */
	d_coords = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * nPoints * nDim, coords, &err);
	OPENCL_CHECK_ERROR(err);
	d_faces = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * nFaces * nVertsPerFace, faces, &err);
	OPENCL_CHECK_ERROR(err);
	d_flowmap = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * nPoints * nDim, flowmap, &err);
	OPENCL_CHECK_ERROR(err);
	d_nFacesPerPoint = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * nPoints, nFacesPerPoint, &err);
	OPENCL_CHECK_ERROR(err);
	d_facesPerPoint = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * nFacesPerPoint[nPoints - 1], facesPerPoint, &err);
	OPENCL_CHECK_ERROR(err);

	/* Allocate additional memory at the device */
	d_logSqrt = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(double) * nPoints, NULL, &err);
	OPENCL_CHECK_ERROR(err);

	/* Solve FTLE */
	OPENCL_CHECK(clFinish(queue));
	gettimeofday(&ftle_clock, NULL);
	kernel_begin = ftle_clock;

	/* STEP 1: compute gradient, tensors and ATxA based on neighbors flowmap values */

	if (nDim == 2)
		fpga_compute_gradient = fpga_compute_gradient_2D;
	else
		fpga_compute_gradient = fpga_compute_gradient_3D;

	OPENCL_CHECK(clSetKernelArg(fpga_compute_gradient, 0, sizeof(cl_int), &nPoints));
	OPENCL_CHECK(clSetKernelArg(fpga_compute_gradient, 1, sizeof(cl_int), &nVertsPerFace));
	OPENCL_CHECK(clSetKernelArg(fpga_compute_gradient, 2, sizeof(cl_mem), &d_coords));
	OPENCL_CHECK(clSetKernelArg(fpga_compute_gradient, 3, sizeof(cl_mem), &d_flowmap));
	OPENCL_CHECK(clSetKernelArg(fpga_compute_gradient, 4, sizeof(cl_mem), &d_faces));
	OPENCL_CHECK(clSetKernelArg(fpga_compute_gradient, 5, sizeof(cl_mem), &d_nFacesPerPoint));
	OPENCL_CHECK(clSetKernelArg(fpga_compute_gradient, 6, sizeof(cl_mem), &d_facesPerPoint));
	OPENCL_CHECK(clSetKernelArg(fpga_compute_gradient, 7, sizeof(cl_mem), &d_logSqrt));
	OPENCL_CHECK(clSetKernelArg(fpga_compute_gradient, 8, sizeof(cl_double), &t_eval));

	OPENCL_CHECK(clEnqueueTask(queue, fpga_compute_gradient, 0, NULL, &kernel_event));

	OPENCL_CHECK(clFinish(queue)); // For timing the kernel

	gettimeofday(&fpga2cpu_transfer_begin, NULL);
	OPENCL_CHECK(clEnqueueReadBuffer(queue, d_logSqrt, CL_TRUE, 0, sizeof(double) * nPoints, logSqrt, 1, &kernel_event, NULL));

	OPENCL_CHECK(clFinish(queue));
	gettimeofday(&fpga_end, NULL);

	/* Device-related execution time */
	// With gettimeofday
	float time_cpu2fpga = (kernel_begin.tv_sec - cpu2fpga_transfer_begin.tv_sec) * 1000 + (kernel_begin.tv_usec - cpu2fpga_transfer_begin.tv_usec) / 1000.0;
	float time_kernel = (fpga2cpu_transfer_begin.tv_sec - kernel_begin.tv_sec) * 1000 + (fpga2cpu_transfer_begin.tv_usec - kernel_begin.tv_usec) / 1000.0;
	float time_fpga2cpu = (fpga_end.tv_sec - fpga2cpu_transfer_begin.tv_sec) * 1000 + (fpga_end.tv_usec - fpga2cpu_transfer_begin.tv_usec) / 1000.0;

	printf("\n=== gettimeofday times ===\nCPU->FPGA transfer time: %f (ms), kernel time: %f (ms), FPGA->CPU time: %f (ms)\n", time_cpu2fpga, time_kernel, time_fpga2cpu);

	/* With OpenCL profiling */
	cl_ulong cl_time_kernel_begin, cl_time_kernel_end;
	OPENCL_CHECK(clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &cl_time_kernel_begin, NULL));
	OPENCL_CHECK(clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &cl_time_kernel_end, NULL));
	printf("=== OpenCL profiling times ===\nKernel time: %f (ms)\n", (cl_time_kernel_end - cl_time_kernel_begin) / 1e6);

	/* Times */
	end_clock = fpga_end;
	time_total = (end_clock.tv_sec - preproc_clock.tv_sec) + (end_clock.tv_usec - ftle_clock.tv_usec) / 1e6;
	time_ftle = (end_clock.tv_sec - ftle_clock.tv_sec) + (end_clock.tv_usec - ftle_clock.tv_usec) / 1e6;

	/* Free memory */
	OPENCL_CHECK(clReleaseMemObject(d_faces));
	OPENCL_CHECK(clReleaseMemObject(d_nFacesPerPoint));
	OPENCL_CHECK(clReleaseMemObject(d_facesPerPoint));
	OPENCL_CHECK(clReleaseMemObject(d_coords));
	OPENCL_CHECK(clReleaseMemObject(d_flowmap));
	OPENCL_CHECK(clReleaseMemObject(d_logSqrt));

	OPENCL_CHECK(clReleaseKernel(fpga_compute_gradient_2D));
	OPENCL_CHECK(clReleaseKernel(fpga_compute_gradient_3D));
	OPENCL_CHECK(clReleaseCommandQueue(queue));
	OPENCL_CHECK(clReleaseContext(ctx));

	printf("DONE\n\n");
	printf("--------------------------------------------------------\n");
	fflush(stdout);

	/* Write result in output file (if desired) */
	if (atoi(argv[6]))
	{
		printf("\nWriting result in output file...                  ");
		fflush(stdout);
		FILE *fp_w = fopen("result_FTLE.csv", "w");
		for (int ii = 0; ii < nPoints; ii++)
			fprintf(fp_w, "%f\n", logSqrt[ii]);
		fclose(fp_w);
		printf("DONE\n\n");
		printf("--------------------------------------------------------\n");
		fflush(stdout);
	}

	/* Show execution time */
	printf("\nExecution time (ms): %f\n\n", time_total * 1000);
	printf("\nPreprocessing time (ms): %f\n\n", (time_total - time_ftle) * 1000);
	printf("\nFTLE time (ms): %f\n\n", time_ftle * 1000);
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
