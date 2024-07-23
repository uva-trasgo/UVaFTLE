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
#include <sycl/sycl.hpp>

using namespace sycl;
#ifndef STRUCTS_H
#define STRUCTS_H
typedef struct Face {
   int      index;      /* index to mesh->faces structure */
   int     *vertices;   /* index to mesh->points structure */
} face_t;

typedef struct Point {
   int       index; // Point index in mesh->points
   double   *coordinates;
   int       nfaces;
   int      *faces;
} point_t;

typedef struct Mesh {
   int       nDim;
   int       nPoints;
   int       nFaces;
   int       nVertsPerFace;
   point_t  *points;
   face_t   *faces;
} mesh_t;
#endif

void read_coordinates ( char *filename, int nDim, int npoints, double *coords );
void read_faces ( char *filename, int nDim, int nVertsPerFace, int nfaces, int *faces );
void read_flowmap ( char *filename, int nDims, int nPoints, double *flowmap );
void read_nFacesPerPoint(char *filename, int nDims, int nPoints, int *nFacesperPoint);
void read_facesPerPoint(char *filename, int nDims, int nElems, int *facesperPoint);
void create_nFacesPerPoint_vector ( int nDim, int nPoints, int nFaces, int nVertsPerFace, int *faces, int *nFacesPerPoint );
//sycl::event create_facesPerPoint_vector (queue* q, int nDim, int nPoints, int nFaces, int nVertsPerFace, sycl::buffer<int, 1> *b_faces, sycl::buffer<int, 1> *b_nFacesPerPoint, sycl::buffer<int, 1> *b_facesPerPoint );
void create_facesPerPoint_vector(int nDim, int nPoints, int nFaces, int nVertsPerFace, int *faces, int *nFacesPerPoint, int *facesPerPoint);
