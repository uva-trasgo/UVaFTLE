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

#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include <vector>

#include <sys/time.h>
#include <time.h>

#include "arithmetic.h"
#include "ftle.h"
#include "preprocess.h"

#define blockSize 512
#define D1_RANGE(size) \
	range<1> { static_cast<size_t>(size) }

using namespace sycl;

float getKernelExecutionTime(sycl::event event)
{
	auto start_time = event.get_profiling_info<sycl::info::event_profiling::command_start>();
	auto end_time = event.get_profiling_info<sycl::info::event_profiling::command_end>();
	return (end_time - start_time) / 1000000.0f;
}

int main(int argc, char *argv[])
{

	printf("--------------------------------------------------------\n");
	printf("|                        UVaFTLE                       |\n");
	printf("|                                                      |\n");
	printf("| Developers:                                          |\n");
	printf("|  - Rocío Carratalá-Sáez | rcarrata@uji.es            |\n");
	printf("|  - Yuri Torres          | yuri.torres@infor.uva.es   |\n");
	printf("|  - Sergio López-Huguet  | serlohu@upv.es             |\n");
	printf("--------------------------------------------------------\n");
	fflush(stdout);

	// Check usage
	if (argc != 7 || argc != 9)
	{
		printf("USAGE: %s <nDim> <coords_file> <faces_file> <flowmap_file> <t_eval> <print2file> [nFaces_file, faces_file]\n", argv[0]);
		printf("\texecutable:    compute_ftle\n");
		printf("\tnDim:    dimensions of the space (2D/3D)\n");
		printf("\tcoords_file:   file where mesh coordinates are stored.\n");
		printf("\tfaces_file:    file where mesh faces are stored.\n");
		printf("\tflowmap_file:  file where flowmap values are stored.\n");
		printf("\tt_eval:        time when compute ftle is desired.\n");
		printf("\tprint to file? (0-NO, 1-YES)\n");
		printf("\t[nFaces_file]: nFaces_per_point precomputed file.\n");
		printf("\t[faces_file]:  faces_per_point precomputed file.\n");
		return 1;
	}

	double t_eval = atof(argv[5]);
	int check_EOF;
	char buffer[255];

	int nDim, nVertsPerFace, nPoints, nFaces;

	double *coords, *flowmap;
	int *faces;
	int *nFacesPerPoint;
	int *facesPerPoint;

	double *w;
	double *logSqrt;

	/* Initialize mesh original information */
	nDim = atoi(argv[1]);
	if (nDim == 2)
		nVertsPerFace = 3; // 2D: faces are triangles
	else if (nDim == 3)
		nVertsPerFace = 4; // 3D: faces (volumes) are tetrahedrons
	else
	{
		printf("Wrong dimension provided (2 or 3 supported)\n");
		return 1;
	}

	/* Read coordinates, faces and flowmap from Python-generated files and generate corresponding GPU vectors */
	/* Read coordinates information */
	printf("\nReading input data\n\n");
	fflush(stdout);
	printf("\tReading mesh points coordinates...        ");
	fflush(stdout);
	FILE *file = fopen(argv[2], "r");
	check_EOF = fscanf(file, "%s", buffer);
	if (check_EOF == EOF)
	{
		fprintf(stderr, "Error: Unexpected EOF in read_coordinates\n");
		fflush(stdout);
		exit(-1);
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
		exit(-1);
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
	fflush(stdout);

	/* Allocate additional memory at the CPU */
	logSqrt = (double *)malloc(sizeof(double) * nPoints);
	nFacesPerPoint = (int *)malloc(sizeof(int) * nPoints); /* REMARK: nFacesPerPoint accumulates previous nFacesPerPoint */

	/* PREPROCESS */
	if (argc == 7)
	{
		/* Assign faces to vertices and generate nFacesPerPoint and facesPerPoint GPU vectors */
		create_nFacesPerPoint_vector(nDim, nPoints, nFaces, nVertsPerFace, faces, nFacesPerPoint);
		facesPerPoint = (int *)malloc(sizeof(int) * nFacesPerPoint[nPoints - 1]);
		create_facesPerPoint_vector(nDim, nPoints, nFaces, nVertsPerFace, faces, nFacesPerPoint, facesPerPoint);
	}
	else // (argc == 9)
	{
		read_nFacesPerPoint(argv[7], nDim, nPoints, nFacesPerPoint);
		facesPerPoint = (int *)malloc(sizeof(int) * nFacesPerPoint[nPoints - 1]);
		read_facesPerPoint(argv[8], nDim, nFacesPerPoint[nPoints - 1], facesPerPoint);
	}

	/* Solve FTLE */
	fflush(stdout);
	auto property_list = sycl::property_list{sycl::property::queue::enable_profiling()};
	/*Generate SYCL queues*/
#ifdef EMULATION
	ext::intel::fpga_emulator_selector d_selector;
#else
	ext::intel::fpga_selector d_selector;
#endif
	queue s_queue(d_selector, property_list);

	printf("Selected device: %s\n", s_queue.get_device().get_info<info::device::name>().c_str());
	std::vector<sycl::event> event_list(2);

	printf("\nComputing FTLE (SYCL)...                     ");
	struct timeval time_kernel_begin, time_kernel_end;
	{
		/*Creating SYCL BUFFERS*/
		sycl::buffer<double, 1> b_coords(coords, D1_RANGE(nPoints * nDim));    // check
		sycl::buffer<int, 1> b_faces(faces, D1_RANGE(nFaces * nVertsPerFace)); // check
		sycl::buffer<double, 1> b_flowmap(flowmap, D1_RANGE(nPoints * nDim));  // check
		sycl::buffer<double, 1> b_logSqrt(logSqrt, D1_RANGE(nPoints));
		sycl::buffer<int, 1> b_nFacesPerPoint(nFacesPerPoint, D1_RANGE(nPoints));                   // check
		sycl::buffer<int, 1> b_facesPerPoint(facesPerPoint, D1_RANGE(nFacesPerPoint[nPoints - 1])); // check

		/* Compute gradient, tensors and ATxA based on neighbors flowmap values, then get the max eigenvalue */
		gettimeofday(&time_kernel_begin, NULL);
		if (nDim == 2)
			event_list[1] = compute_gradient_2D(&s_queue, nPoints, nVertsPerFace, &b_coords, &b_flowmap, &b_faces, &b_nFacesPerPoint, &b_facesPerPoint, &b_logSqrt, t_eval);
		else
			event_list[1] = compute_gradient_3D(&s_queue, nPoints, nVertsPerFace, &b_coords, &b_flowmap, &b_faces, &b_nFacesPerPoint, &b_facesPerPoint, &b_logSqrt, t_eval);
		gettimeofday(&time_kernel_end, NULL);
	}
	/* Time */
	printf("DONE\n\n");
	printf("--------------------------------------------------------\n");
	printf("FTLE kernel time: %f (ms)\n", (time_kernel_end.tv_sec - time_kernel_begin.tv_sec) * 1000 + (time_kernel_end.tv_usec - time_kernel_begin.tv_usec) / 1000.0);
	fflush(stdout);

	/* Print numerical results */
	if (atoi(argv[6]))
	{
		printf("\nWriting result in output file...                  ");
		fflush(stdout);
		FILE *fp_w = fopen("ftle_result.csv", "w");
		for (int ii = 0; ii < nPoints; ii++)
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
	printf("Execution times in miliseconds\n");
	printf("FTLE kernel: %f\n", getKernelExecutionTime(event_list[1]));
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
