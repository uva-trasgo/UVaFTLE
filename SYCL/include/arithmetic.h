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
#include "ftle.h"

event compute_gradient_2D (queue* q,  int nPoints, int offset, int faces_offset, int nVertsPerFace, buffer<double, 1> *b_coords, buffer<double, 1> *b_flowmap, buffer<int, 1> *b_faces, buffer<int, 1> *b_nFacesPerPoint, buffer<int, 1> *b_facesPerPoint, buffer<double, 1> *b_log_sqrt, double T );
event compute_gradient_3D (queue* q,  int nPoints, int offset, int faces_offset, int nVertsPerFace, buffer<double, 1> *b_coords, buffer<double, 1> *b_flowmap, buffer<int, 1> *b_faces, buffer<int, 1> *b_nFacesPerPoint, buffer<int, 1> *b_facesPerPoint, buffer<double, 1> *b_log_sqrt, double T );
