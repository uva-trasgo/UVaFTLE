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

////////////////////////////////////////////////// 
////////////////////////////////////////////////// 
//                                              //
//  File:   ftle_2D_reference.cpp               //
//                                              //
//  Authors:                                    //
//          Roberto Rodríguez Osorio            //
//          University of A Coruña, Spain       //
//          ORCID:  0000-0001-8768-2240         //
//          Manuel de Castro Caballero          //
//          University of Valladolid, Spain     //
//          ORCID:  0000-0003-3080-5136         //
//                                              //
////////////////////////////////////////////////// 
////////////////////////////////////////////////// 

#include <iostream>
#include "ap_int.h"
#include "math.h"

using namespace std;

// double arithmetic3D() performs the actual calculations for the 3D case
// The 4 required coordinates: X0, X1, Y2, Y3, Z4 and, Z5 are provided by the caller function,
// as well as the 18 flow values 

double arithmetic3D(double X0,
		double X1,
		double Y2,
		double Y3,
		double Z4,
		double Z5,
		double flowX0,
		double flowX1,
		double flowX2,
		double flowX3,
		double flowX4,
		double flowX5,
		double flowY0,
		double flowY1,
		double flowY2,
		double flowY3,
		double flowY4,
		double flowY5,
		double flowZ0,
		double flowZ1,
		double flowZ2,
		double flowZ3,
		double flowZ4,
		double flowZ5,
		const double factorT
){
#pragma HLS INTERFACE ap_ctrl_none port=X0
#pragma HLS INTERFACE ap_none port=X0
#pragma HLS INTERFACE ap_ctrl_none port=X1
#pragma HLS INTERFACE ap_none port=X1
#pragma HLS INTERFACE ap_ctrl_none port=Y2
#pragma HLS INTERFACE ap_none port=Y2
#pragma HLS INTERFACE ap_ctrl_none port=Y3
#pragma HLS INTERFACE ap_none port=Y3
#pragma HLS INTERFACE ap_ctrl_none port=Z4
#pragma HLS INTERFACE ap_none port=Z4
#pragma HLS INTERFACE ap_ctrl_none port=Z5
#pragma HLS INTERFACE ap_none port=Z5

#pragma HLS INTERFACE ap_ctrl_none port=flowX0
#pragma HLS INTERFACE ap_none port=flowX0
#pragma HLS INTERFACE ap_ctrl_none port=flowY0
#pragma HLS INTERFACE ap_none port=flowY0
#pragma HLS INTERFACE ap_ctrl_none port=flowZ0
#pragma HLS INTERFACE ap_none port=flowZ0
#pragma HLS INTERFACE ap_ctrl_none port=flowX1
#pragma HLS INTERFACE ap_none port=flowX1
#pragma HLS INTERFACE ap_ctrl_none port=flowY1
#pragma HLS INTERFACE ap_none port=flowY1
#pragma HLS INTERFACE ap_ctrl_none port=flowZ1
#pragma HLS INTERFACE ap_none port=flowZ1
#pragma HLS INTERFACE ap_ctrl_none port=flowX2
#pragma HLS INTERFACE ap_none port=flowX2
#pragma HLS INTERFACE ap_ctrl_none port=flowY2
#pragma HLS INTERFACE ap_none port=flowY2
#pragma HLS INTERFACE ap_ctrl_none port=flowZ2
#pragma HLS INTERFACE ap_none port=flowZ2
#pragma HLS INTERFACE ap_ctrl_none port=flowX3
#pragma HLS INTERFACE ap_none port=flowX3
#pragma HLS INTERFACE ap_ctrl_none port=flowY3
#pragma HLS INTERFACE ap_none port=flowY3
#pragma HLS INTERFACE ap_ctrl_none port=flowZ3
#pragma HLS INTERFACE ap_none port=flowZ3
#pragma HLS INTERFACE ap_ctrl_none port=flowX4
#pragma HLS INTERFACE ap_none port=flowX4
#pragma HLS INTERFACE ap_ctrl_none port=flowY4
#pragma HLS INTERFACE ap_none port=flowY4
#pragma HLS INTERFACE ap_ctrl_none port=flowZ4
#pragma HLS INTERFACE ap_none port=flowZ4
#pragma HLS INTERFACE ap_ctrl_none port=flowX5
#pragma HLS INTERFACE ap_none port=flowX5
#pragma HLS INTERFACE ap_ctrl_none port=flowY5
#pragma HLS INTERFACE ap_none port=flowY5
#pragma HLS INTERFACE ap_ctrl_none port=flowZ5
#pragma HLS INTERFACE ap_none port=flowZ5

#pragma HLS pipeline II = 1

    double denom_x = (X1 - X0);
    double denom_y = (Y3 - Y2);
    double denom_z = (Z5 - Z4);

    double gra10 = (flowX1 - flowX0) / denom_x;
    double gra11 = (flowX3 - flowX2) / denom_y;
    double gra12 = (flowX5 - flowX4) / denom_z;

    double gra20 = (flowY1 - flowY0) / denom_x;
    double gra21 = (flowY3 - flowY2) / denom_y;
    double gra22 = (flowY5 - flowY4) / denom_z;

    double gra30 = (flowZ1 - flowZ0) / denom_x;
    double gra31 = (flowZ3 - flowZ2) / denom_y;
    double gra32 = (flowZ5 - flowZ4) / denom_z;



    double fra10 = gra10 * gra10 + gra20 * gra20 + gra30 * gra30;
    double fra11 = gra10 * gra11 + gra20 * gra21 + gra30 * gra31;
    double fra12 = gra10 * gra12 + gra20 * gra22 + gra30 * gra32;
    double fra20 = fra11;
    double fra21 = gra11 * gra11 + gra21 * gra21 + gra31 * gra31;
    double fra22 = gra11 * gra12 + gra11 * gra22 + gra31 * gra32;
    double fra30 = fra12;
    double fra31 = fra22;
    double fra32 = gra12 * gra12 + gra22 * gra22 + gra32 * gra32;

    double A10 = fra10 * fra10 + fra11 * fra11 + fra12 * fra12;
    double A11 = fra10 * fra20 + fra11 * fra21 + fra12 * fra22;
    double A12 = fra10 * fra30 + fra11 * fra31 + fra12 * fra32;
    double A20 = fra20 * fra10 + fra21 * fra11 + fra22 * fra12;
    double A21 = fra20 * fra20 + fra21 * fra21 + fra22 * fra22;
    double A22 = fra20 * fra30 + fra21 * fra31 + fra22 * fra32;
    double A30 = fra30 * fra10 + fra31 * fra11 + fra32 * fra12;
    double A31 = fra30 * fra20 + fra31 * fra21 + fra32 * fra22;
    double A32 = fra30 * fra30 + fra31 * fra31 + fra32 * fra32;

    double b = A10 + A21 + A32;
    double c = A12 * A30 + A22 * A31 + A11 * A20 - A10 * A21 - A10 * A32 - A21 * A32;
    double d =
        A10 * A21 * A32 + A11 * A22 * A30 + A12 * A20 * A31 - A10 * A22 * A31 - A11 * A20 * A32 - A12 * A21 * A30;

    double A = b * b + 3 * c;
    double B = b * c + 9 * d;
    double C = c * c - 3 * b * d;
    double del = B * B - 4 * A * C;
	
// The square root is actually calculated from its reciprocal. 
// Therefore, if the reciprocal is going to be used, it is important to make it clear to the compiler. 

    double iA  = -1 / A;
    double iSQA = hls::rsqrt(A);
	double SQA = A * iSQA; 
#pragma HLS RESOURCE variable=iSQA core=DRSqrt

    double x1, x2, x3;

	if (A == B && A == 0)
	 {
		 x1 = x2 = x3 = -b / (3 * a);
	 }
	else if (del == 0)
	 {
		 x1 = -b / a + B / A;
		 x2 = x3 = (-B / A) / 2;
	 }
	else
	 {
		double temp = (b - (1.5 * B * iA)) * iSQA;
		double _xt = acos(temp);
		double xt = _xt * 0.333333333333333333333333333333333;
		double const a3 = -0.333333333333333333333333333333333;
		x1 = (-b - 2 * SQA * cos(xt)) * a3;
		x2 = (-b + SQA * (cos(xt) + sqrt(3) * sin(xt))) * a3;
		x3 = (-b + SQA * (cos(xt) - sqrt(3) * sin(xt))) * a3;
	 }
	 
    double first = x1 >= x2 ? x1 : x2;
    double max = first >= x3 ? first : x3;

	double logarithm = log(max);
	return logarithm * factorT;
	
	// originally: 	return log( sqrt(max) ) / T;
	// as T is constant, factorT is defined as T / 2
	// taking advantage of: log( sqrt(max) ) = 1/2 * log(max)
	// and multiplying by factorT instead of the more costly square root and quotient. 	
}



double ftle_3d(int vec0, int vec1, int vec2, int vec3, int vec4, int vec5, double *coordsX, double *flowmapX, double *coordsY, double *flowmapY, double *coordsZ, double *flowmapZ, double factorT){

#pragma HLS INTERFACE ap_ctrl_none port=vec0
#pragma HLS INTERFACE ap_none port=vec0
#pragma HLS INTERFACE ap_ctrl_none port=vec1
#pragma HLS INTERFACE ap_none port=vec1
#pragma HLS INTERFACE ap_ctrl_none port=vec2
#pragma HLS INTERFACE ap_none port=vec2
#pragma HLS INTERFACE ap_ctrl_none port=vec3
#pragma HLS INTERFACE ap_none port=vec3
#pragma HLS INTERFACE ap_ctrl_none port=vec4
#pragma HLS INTERFACE ap_none port=vec4
#pragma HLS INTERFACE ap_ctrl_none port=vec5
#pragma HLS INTERFACE ap_none port=vec5

#pragma HLS INTERFACE m_axi port = coordsX depth = 1000000000 bundle = gmemCoordsX
#pragma HLS INTERFACE m_axi port = flowmapX depth = 1000000000 bundle = gmemflomapX
#pragma HLS INTERFACE m_axi port = coordsY depth = 1000000000 bundle = gmemCoordsY
#pragma HLS INTERFACE m_axi port = flowmapY depth = 1000000000 bundle = gmemFlowmapY
#pragma HLS INTERFACE m_axi port = coordsZ depth = 1000000000 bundle = gmemCoordsZ
#pragma HLS INTERFACE m_axi port = flowmapZ depth = 1000000000 bundle = gmemFlowmapZ

#pragma HLS pipeline II = 1

	double X0, X1, Y2, Y3, Z4, Z5, flowX0, flowX1, flowX2, flowX3, flowX4, flowX5, flowY0, flowY1, flowY2, flowY3, flowY4, flowY5, flowZ0, flowZ1, flowZ2, flowZ3, flowZ4, flowZ5;

	X0 = vec0 != -1 ? coordsX[vec0] : 0.0;
	X1 = vec1 != -1 ? coordsX[vec1] : 0.0;
	Y2 = vec2 != -1 ? coordsY[vec2] : 0.0;
	Y3 = vec3 != -1 ? coordsY[vec3] : 0.0;
	Z4 = vec4 != -1 ? coordsZ[vec4] : 0.0;
	Z5 = vec5 != -1 ? coordsZ[vec5] : 0.0;

	flowX0 = vec0 != -1 ? flowmapX[vec0] : 0.0;
	flowX1 = vec1 != -1 ? flowmapX[vec1] : 0.0;
	flowX2 = vec2 != -1 ? flowmapX[vec2] : 0.0;
	flowX3 = vec3 != -1 ? flowmapX[vec3] : 0.0;
	flowX4 = vec4 != -1 ? flowmapX[vec5] : 0.0;
	flowX5 = vec5 != -1 ? flowmapX[vec4] : 0.0;

	flowY0 = vec0 != -1 ? flowmapY[vec0] : 0.0;
	flowY1 = vec1 != -1 ? flowmapY[vec1] : 0.0;
	flowY2 = vec2 != -1 ? flowmapY[vec2] : 0.0;
	flowY3 = vec3 != -1 ? flowmapY[vec3] : 0.0;
	flowY4 = vec4 != -1 ? flowmapY[vec4] : 0.0;
	flowY5 = vec5 != -1 ? flowmapY[vec5] : 0.0;

	flowZ0 = vec0 != -1 ? flowmapZ[vec0] : 0.0;
	flowZ1 = vec1 != -1 ? flowmapZ[vec1] : 0.0;
	flowZ2 = vec2 != -1 ? flowmapZ[vec2] : 0.0;
	flowZ3 = vec3 != -1 ? flowmapZ[vec3] : 0.0;
	flowZ4 = vec4 != -1 ? flowmapZ[vec4] : 0.0;
	flowZ5 = vec5 != -1 ? flowmapZ[vec5] : 0.0;

	return arithmetic3D(X0, X1, Y2, Y3, Z4, Z5, flowX0, flowX1, flowX2, flowX3, flowX4, flowX5, flowY0, flowY1, flowY2, flowY3, flowY4, flowY5, flowZ0, flowZ1, flowZ2, flowZ3, flowZ4, flowZ5, factorT);
}

