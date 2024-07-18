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
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <assert.h>
#include "omp.h"

#include "ftle.h"
#include "arithmetic.h"
#include "preprocess.h"

#define blockSize 512

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
		printf("USAGE: %s <nDim> <coords_file> <faces_file> <flowmap_file> <t_eval> <nth> <print2file>\n", argv[0]);
		printf("\texecutable:    compute_ftle\n");
		printf("\tnDim:    dimensions of the space (2D/3D)\n");
		printf("\tcoords_file:   file where mesh coordinates are stored.\n");
		printf("\tfaces_file:    file where mesh faces are stored.\n");
		printf("\tflowmap_file:  file where flowmap values are stored.\n");
		printf("\tt_eval:        time when compute ftle is desired.\n");
		printf("\tnth:           number of OpenMP threads to use.\n");
		printf("\tprint to file? (0-NO, 1-YES)\n");
		return 1;
	}

	struct timeval preproc_clock;
	struct timeval ftle_clock;
	struct timeval end_clock;
	double time;
	double t_eval = atof(argv[5]);
	int nth, check_EOF;
	char buffer[255];

	int nDim, nVertsPerFace, nPoints, nFaces;

	double *coords, *flowmap;
	int    *faces, *d2_faces;
	int    *nFacesPerPoint, *d2_nFacesPerPoint;
	int    *facesPerPoint, *d2_facesPerPoint;

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
    gettimeofday(&preproc_clock, NULL);
#ifdef DYNAMIC
    printf("\nComputing Preproc(dynamic scheduler)...                     ");
    #pragma omp parallel for default(none) shared(nDim, nFaces, nPoints, nVertsPerFace, faces, nFacesPerPoint,  facesPerPoint) num_threads(nth) schedule(dynamic)
#elif defined GUIDED
    printf("\nComputing Preproc (guided scheduler)...                     ");
    #pragma omp parallel for default(none) shared(nDim, nFaces, nPoints, nVertsPerFace, faces, nFacesPerPoint,  facesPerPoint) num_threads(nth) schedule(guided)
#else
    printf("\nComputing Preproc (static scheduler)...                     ");
    #pragma omp parallel for default(none) shared(nDim, nFaces, nPoints, nVertsPerFace, faces, nFacesPerPoint,  facesPerPoint) num_threads(nth) schedule(static)
#endif
	for ( int ip = 0; ip < nPoints; ip++ )
             create_facesPerPoint_vector( nDim, ip, nFaces, nVertsPerFace, faces, nFacesPerPoint, facesPerPoint );    

    /* Solve FTLE */
    nth = atoi(argv[6]);
    fflush(stdout);
	gettimeofday(&ftle_clock, NULL);

#ifdef DYNAMIC
    printf("\nComputing FTLE (dynamic scheduler)...                     ");
    #pragma omp parallel for default(none) shared(nDim, nPoints, nFaces, nVertsPerFace, coords, flowmap, faces, nFacesPerPoint, facesPerPoint, logSqrt, t_eval) num_threads(nth) schedule(dynamic)
#elif defined GUIDED
    printf("\nComputing FTLE (guided scheduler)...                     ");
    #pragma omp parallel for default(none) shared(nDim, nPoints, nFaces, nVertsPerFace, coords, flowmap, faces, nFacesPerPoint, facesPerPoint,  logSqrt, t_eval) num_threads(nth) schedule(guided)
#else
     printf("\nComputing FTLE (static scheduler)...                     ");
    #pragma omp parallel for default(none) shared(nDim, nPoints, nFaces, nVertsPerFace, coords, flowmap, faces, nFacesPerPoint, facesPerPoint, logSqrt, t_eval) num_threads(nth) schedule(static)
#endif
	for ( int ip = 0; ip < nPoints; ip++ )
	{
    	/* Compute gradient, tensors and ATxA based on neighbors flowmap values, then get the max eigenvalue */
		if ( nDim == 2 )
			compute_gradient_2D ( ip, nVertsPerFace, 
				coords, flowmap, faces, nFacesPerPoint, facesPerPoint, 
				logSqrt, t_eval);
		else
			compute_gradient_3D  ( ip, nVertsPerFace, 
				coords, flowmap, faces, nFacesPerPoint, facesPerPoint, 
				logSqrt, t_eval);
	}
   
   	/* Time */
	gettimeofday(&end_clock, NULL);
	printf("DONE\n\n");
	printf("--------------------------------------------------------\n");
    fflush(stdout);

    /* Print numerical results */
	if ( atoi(argv[7]) )
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
    time = (ftle_clock.tv_sec - preproc_clock.tv_sec) + (ftle_clock.tv_usec - preproc_clock.tv_usec)/1000000.0;
	printf("\nExecution time (ms) with %d threads: %f\n\n", nth, time*1000);
	time = (end_clock.tv_sec - ftle_clock.tv_sec) + (end_clock.tv_usec - ftle_clock.tv_usec)/1000000.0;
	printf("\nExecution time (ms) with %d threads: %f\n\n", nth, time*1000);
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
