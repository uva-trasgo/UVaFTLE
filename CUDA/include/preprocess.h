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
#include "ftle.h"

void read_coordinates ( char *filename, int nDim, int npoints, double *coords );
void read_faces ( char *filename, int nDim, int nVertsPerFace, int nfaces, int *faces );
void read_flowmap ( char *filename, int nDims, int nPoints, double *flowmap );
void create_nFacesPerPoint_vector ( int nDim, int nPoints, int nFaces, int nVertsPerFace, int *faces, int *nFacesPerPoint );
__global__ void create_facesPerPoint_vector (int nDim, int nPoints, int offset, int faces_offset, int nFaces, int nVertsPerFace, int *faces, int *nFacesPerPoint, int *facesPerPoint );

