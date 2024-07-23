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
#include <vector>
#include <iostream>

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

sycl::event compute_gradient_2D (queue* q,  int nPoints, int nVertsPerFace, sycl::buffer<double, 1> *b_coords, sycl::buffer<double, 1> *b_flowmap, sycl::buffer<int, 1> *b_faces, 
					sycl::buffer<int, 1> *b_nFacesPerPoint, sycl::buffer<int, 1> *b_facesPerPoint, sycl::buffer<double, 1> *b_log_sqrt, double T );
sycl::event compute_gradient_3D (queue* q,  int nPoints, int nVertsPerFace, sycl::buffer<double, 1> *b_coords, sycl::buffer<double, 1> *b_flowmap, sycl::buffer<int, 1> *b_faces, 
					sycl::buffer<int, 1> *b_nFacesPerPoint, sycl::buffer<int, 1> *b_facesPerPoint, sycl::buffer<double, 1> *b_log_sqrt, double T );
double log_sqrt ( double T, double eigen );
double max_solve_3rd_degree_eq ( double a, double b, double c, double d);
double max_eigen_2D ( double A10, double A11, double A20, double A21 );
double max_eigen_3D ( double A10, double A11, double A12,
                double A20, double A21, double A22,
                double A30, double A31, double A32 );

