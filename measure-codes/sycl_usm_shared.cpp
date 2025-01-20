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
#include <sys/time.h>
#include <sycl/sycl.hpp>

using namespace sycl;

void read_coordinates ( char *filename, int nDim, int nPoints, double *coords )
{
 int ip, d, check_EOF;
 char buffer[255];
 FILE *file;


 file = fopen( filename, "r" );


 check_EOF = fscanf(file, "%s", buffer);
 if ( check_EOF == EOF )
 {
  fprintf( stderr, "Error: Unexpected EOF in read_coordinates\n" );
  exit(-1);
 }


 for ( ip = 0; ip < nPoints; ip++ )
 {
  for ( d = 0; d < nDim; d++ )
  {
   check_EOF = fscanf(file, "%s", buffer);
   if ( check_EOF == EOF )
   {
    fprintf( stderr, "Error: Unexpected EOF in read_coordinates\n" );
    exit(-1);
   }
   coords[ip * nDim + d] = atof(buffer);
  }
 }


 fclose(file);
}

void read_faces ( char *filename, int nDim, int nVertsPerFace, int nFaces, int *faces )
{
   int iface, ielem, check_EOF;
   char buffer[255];
   FILE *file;


   file = fopen( filename, "r" );


   check_EOF = fscanf(file, "%s", buffer);
   if ( check_EOF == EOF )
   {
      fprintf( stderr, "Error: Unexpected EOF in read_faces\n" );
      exit(-1);
   }


   for ( iface = 0; iface < nFaces; iface++ )
   {
      for ( ielem = 0; ielem < nVertsPerFace; ielem++ )
      {
  check_EOF = fscanf(file, "%s", buffer);
  if ( check_EOF == EOF )
  {
     fprintf( stderr, "Error: Unexpected EOF in read_faces\n" );
     exit(-1);
  }
  faces[iface * nVertsPerFace + ielem] = atoi(buffer);
      }
   }


   fclose(file);
}

void read_flowmap ( char *filename, int nDims, int nPoints, double *flowmap )
{
   int ip, idim, check_EOF;
   char buffer[255];
   FILE *file;


   file = fopen( filename, "r" );


   for ( ip = 0; ip < nPoints; ip++ )
   {
      for ( idim = 0; idim < nDims; idim++ )
   {
  check_EOF = fscanf(file, "%s", buffer);
  if ( check_EOF == EOF )
  {
   fprintf( stderr, "Error: Unexpected EOF in read_flowmap\n" );
   exit(-1);
  }
  flowmap[ip * nDims + idim] = atof(buffer);
   }
   }


   fclose(file);
}

void create_nFacesPerPoint_vector ( int nDim, int nPoints, int nFaces, int nVertsPerFace, int *faces, int *nFacesPerPoint )
{
 int ip, iface, ipf;
 for ( ip = 0; ip < nPoints; ip++ )
 {
  nFacesPerPoint[ip] = 0;
 }
 for ( iface = 0; iface < nFaces; iface++ )
 {
  for ( ipf = 0; ipf < nVertsPerFace; ipf++ )
  {
   ip = faces[iface * nVertsPerFace + ipf];
   nFacesPerPoint[ip] = nFacesPerPoint[ip] + 1;
  }
 }
 for ( ip = 1; ip < nPoints; ip++ )
 {
  nFacesPerPoint[ip] = nFacesPerPoint[ip] + nFacesPerPoint[ip-1];
 }
}

event create_facesPerPoint_vector (queue* q, int nDim, int nPoints, int offset, int faces_offset, int nFaces, int nVertsPerFace, int* faces, int* nFacesPerPoint, int* facesPerPoint)
{
int size = (nPoints% 512) ? (nPoints/512 +1)*512: nPoints;
return q->parallel_for<class preprocess> (nd_range<1>(range<1>{static_cast<size_t>(size)},range<1>{static_cast<size_t>(512)}), [=](nd_item<1> i){
 if(i.get_global_id(0) < nPoints){
  int th_id = i.get_global_id(0) + offset;
  int count, iface, ipf, nFacesP, iFacesP;
  count = 0;
  iFacesP = (( th_id == 0 ) ? 0 : nFacesPerPoint[th_id-1]) - faces_offset;
  nFacesP = ( th_id == 0 ) ? nFacesPerPoint[th_id] : nFacesPerPoint[th_id] - nFacesPerPoint[th_id-1];
  for ( iface = 0; ( iface < nFaces ) && ( count < nFacesP ); iface++ ){
   for ( ipf = 0; ipf < nVertsPerFace; ipf++ ){
    if ( faces[iface * nVertsPerFace + ipf] == th_id ){
     facesPerPoint[iFacesP + count] = iface;
     count++;
    }
   }
  }
 }});
}

event compute_gradient_2D (queue* q, int nPoints, int offset, int faces_offset, int nVertsPerFace, double* coords, double* flowmap, int* faces, int* nFacesPerPoint, int* facesPerPoint, double* logSqrt, double T )
{
 int size = (nPoints% 512) ? (nPoints/512 +1)*512: nPoints;
return q->parallel_for<class ftle2D> (nd_range<1>(range<1>{static_cast<size_t>(size)},range<1>{static_cast<size_t>(512)}), [=](nd_item<1> i){
 if(i.get_global_id(0) < nPoints){
  int th_id = i.get_global_id(0) + offset;
  int nDim = 2;
  int iface, nFaces, idxface, ivert;
  int closest_points_0 = -1;
  int closest_points_1 = -1;
  int closest_points_2 = -1;
  int closest_points_3 = -1;
  int count = 0;

  int ivertex;
  double denom_x, denom_y;
  double gra10, gra11, gra20, gra21;
  double ftle_matrix[4], d_W_ei[2];
  nFaces = (th_id == 0) ? nFacesPerPoint[th_id] : nFacesPerPoint[th_id] - nFacesPerPoint[th_id-1];

  int base_face = (th_id == 0) ? 0 : nFacesPerPoint[th_id-1] - faces_offset;
  for ( iface = 0; (iface < nFaces) && (count < 4); iface++ )
  {
   idxface = facesPerPoint[base_face + iface];
   for ( ivert = 0; (ivert < nVertsPerFace) && (count < 4); ivert++ )
   {
    ivertex = faces[idxface * nVertsPerFace + ivert];
    if ( ivertex != th_id )
    {

     if ( (coords[ivertex * nDim + 1] == coords[th_id * nDim + 1]) && (coords[ivertex * nDim] < coords[th_id * nDim]) )
     {
      if (closest_points_0 == -1)
      {
       closest_points_0 = ivertex;
       count++;
      }
     }
     else
     {

      if ( (coords[ivertex * nDim + 1] == coords[th_id * nDim + 1]) && (coords[ivertex * nDim] > coords[th_id * nDim]) )
      {
       if (closest_points_1 == -1)
       {
        closest_points_1 = ivertex;
        count++;
       }
      }
      else
      {

       if ( (coords[ivertex * nDim] == coords[th_id * nDim]) && (coords[ivertex * nDim + 1] < coords[th_id * nDim + 1]) )
       {
        if (closest_points_2 == -1)
        {
         closest_points_2 = ivertex;
         count++;
        }
       }
       else
       {

        if ( (coords[ivertex * nDim] == coords[th_id * nDim]) && (coords[ivertex * nDim + 1] > coords[th_id * nDim + 1]) )
        {
         if (closest_points_3 == -1)
         {
          closest_points_3 = ivertex;
          count++;
         }
        }
       }
      }
     }
    }
   }
  }
  if ( count == 4 )
  {

   denom_x = coords[ closest_points_1 * nDim ] - coords[ closest_points_0 * nDim ];
   denom_y = coords[ closest_points_3 * nDim + 1] - coords[ closest_points_2 * nDim + 1];
   gra10 = ( flowmap[ closest_points_1 * nDim ] - flowmap[ closest_points_0 * nDim ] ) / denom_x;
   gra20 = ( flowmap[ closest_points_3 * nDim ] - flowmap[ closest_points_2 * nDim ] ) / denom_y;
   gra11 = ( flowmap[ closest_points_1 * nDim + 1 ] - flowmap[ closest_points_0 * nDim + 1 ] ) / denom_x;
   gra21 = ( flowmap[ closest_points_3 * nDim + 1 ] - flowmap[ closest_points_2 * nDim + 1 ] ) / denom_y;
  }
  else
  {
   if ( count == 3 )
   {

    if ( ( closest_points_0 > -1 ) && ( closest_points_1 > -1 ) )
    {
     denom_x = coords[ closest_points_1 * nDim ] - coords[ closest_points_0 * nDim ];
     gra10 = ( flowmap[ closest_points_1 * nDim ] - flowmap[ closest_points_0 * nDim ] ) / denom_x;
     gra20 = 1;
     gra11 = ( flowmap[ closest_points_1 * nDim + 1 ] - flowmap[ closest_points_0 * nDim + 1 ] ) / denom_x;
     gra21 = 1;
    }

    else
    {
     denom_y = coords[ closest_points_3 * nDim + 1] - coords[ closest_points_2 * nDim + 1];
     gra10 = 1;
     gra20 = ( flowmap[ closest_points_3 * nDim ] - flowmap[ closest_points_2 * nDim ] ) / denom_y;
     gra11 = 1;
     gra21 = ( flowmap[ closest_points_3 * nDim + 1 ] - flowmap[ closest_points_2 * nDim + 1 ] ) / denom_y;
    }
   }
   else
   {
    gra10 = 1;
    gra20 = 1;
    gra11 = 1;
    gra21 = 1;
   }
  }

  ftle_matrix[0] = gra10 * gra10 + gra11 * gra11;
  ftle_matrix[1] = gra10 * gra20 + gra11 * gra21;
  ftle_matrix[2] = gra20 * gra10 + gra21 * gra11;
  ftle_matrix[3] = gra20 * gra20 + gra21 * gra21;
  gra10 = ftle_matrix[0];
  gra11 = ftle_matrix[1];
  gra20 = ftle_matrix[2];
  gra21 = ftle_matrix[3];

  ftle_matrix[0] = gra10 * gra10 + gra11 * gra11;
  ftle_matrix[1] = gra10 * gra20 + gra11 * gra21;
  ftle_matrix[2] = gra20 * gra10 + gra21 * gra11;
  ftle_matrix[3] = gra20 * gra20 + gra21 * gra21;
  double A10 = ftle_matrix[0];
  double A11 = ftle_matrix[1];
  double A20 = ftle_matrix[2];
  double A21 = ftle_matrix[3];
  double sq = sycl::sqrt(A21 * A21 + A10 * A10 - 2 * (A10 * A21) + 4 * (A11 * A20));
  d_W_ei[0] = (A21 + A10 + sq) / 2;
  d_W_ei[1] = (A21 + A10 - sq) / 2;


  double max = d_W_ei[0];
  if (d_W_ei[1] > max ) max = d_W_ei[1];
  max = sycl::sqrt(max);
  max = sycl::log (max);

  logSqrt[i.get_global_id(0)] = max / T;
     }
 });
}

event compute_gradient_3D (queue* q, int nPoints, int offset, int faces_offset, int nVertsPerFace, double* coords, double* flowmap, int* faces, int* nFacesPerPoint, int* facesPerPoint, double* logSqrt, double T )
{
int size = (nPoints% 512) ? (nPoints/512 +1)*512: nPoints;
return q->parallel_for<class ftle3D> (nd_range<1>(range<1>{static_cast<size_t>(size)},range<1>{static_cast<size_t>(512)}), [=](nd_item<1> i){
  if(i.get_global_id(0) < nPoints){
  int th_id = i.get_global_id(0) + offset;
  int nDim = 3;
  int iface, nFaces, idxface, ivert;
  int closest_points_0 = -1;
  int closest_points_1 = -1;
  int closest_points_2 = -1;
  int closest_points_3 = -1;
  int closest_points_4 = -1;
  int closest_points_5 = -1;
  int count = 0;

  int ivertex;
  double denom_x, denom_y, denom_z;
  double ftle_matrix[9];
  double gra10, gra11, gra12, gra20, gra21, gra22, gra30, gra31, gra32;
  nFaces = (th_id == 0) ? nFacesPerPoint[th_id] : nFacesPerPoint[th_id] - nFacesPerPoint[th_id-1];

  int base_face = (th_id == 0) ? 0 : nFacesPerPoint[th_id-1] - faces_offset;
  for ( iface = 0; (iface < nFaces) && (count < 6); iface++ ) {
   idxface = facesPerPoint[base_face + iface];
   for ( ivert = 0; (ivert < nVertsPerFace) && (count < 6); ivert++ )
   {
    ivertex = faces[idxface * nVertsPerFace + ivert];
    if ( ivertex != th_id ){

     if ( (coords[ivertex * nDim + 1] == coords[th_id * nDim + 1])
      && (coords[ivertex * nDim + 2] == coords[th_id * nDim + 2])
      && (coords[ivertex * nDim] < coords[th_id * nDim]) )
     {
      if (closest_points_0 == -1)
      {
       closest_points_0 = ivertex;
       count++;
      }
     }
     else
     {

      if ( (coords[ivertex * nDim + 1] == coords[th_id * nDim + 1])
       && (coords[ivertex * nDim + 2] == coords[th_id * nDim + 2])
       && (coords[ivertex * nDim] > coords[th_id * nDim]) )
      {
       if (closest_points_1 == -1)
       {
        closest_points_1 = ivertex;
        count++;
       }
      }
      else
      {

       if ( (coords[ivertex * nDim] == coords[th_id * nDim])
        && (coords[ivertex * nDim + 2] == coords[th_id * nDim + 2])
        && (coords[ivertex * nDim + 1] < coords[th_id * nDim + 1]) )
       {
        if (closest_points_2 == -1)
        {
         closest_points_2 = ivertex;
         count++;
        }
       }
       else
       {

        if ( (coords[ivertex * nDim] == coords[th_id * nDim])
         && (coords[ivertex * nDim + 2] == coords[th_id * nDim + 2])
         && (coords[ivertex * nDim + 1] > coords[th_id * nDim + 1]) )
        {
         if (closest_points_3 == -1)
         {
          closest_points_3 = ivertex;
          count++;
         }
        }
        else
        {

         if ( (coords[ivertex * nDim] == coords[th_id * nDim])
          && (coords[ivertex * nDim + 1] == coords[th_id * nDim + 1])
          && (coords[ivertex * nDim + 2] < coords[th_id * nDim + 2]) )
         {
          if (closest_points_4 == -1)
          {
           closest_points_4 = ivertex;
           count++;
          }
         }
         else
         {

          if ( (coords[ivertex * nDim] == coords[th_id * nDim])
           && (coords[ivertex * nDim + 1] == coords[th_id * nDim + 1])
           && (coords[ivertex * nDim + 2] > coords[th_id * nDim + 2]) )
          {
           if (closest_points_5 == -1)
           {
            closest_points_5 = ivertex;
            count++;
           }
          }
         }
        }
       }
      }
     }
    }
   }
  }
  if ( count == 6 ){

   denom_x = coords[ closest_points_1 * nDim ] - coords[ closest_points_0 * nDim ];
   denom_y = coords[ closest_points_3 * nDim + 1] - coords[ closest_points_2 * nDim + 1];
   denom_z = coords[ closest_points_5 * nDim + 2] - coords[ closest_points_4 * nDim + 2];

   gra10 = ( flowmap[ closest_points_1 * nDim ] - flowmap[ closest_points_0 * nDim ] ) / denom_x;
   gra11 = ( flowmap[ closest_points_3 * nDim ] - flowmap[ closest_points_2 * nDim ] ) / denom_y;
      gra12 = ( flowmap[ closest_points_5 * nDim ] - flowmap[ closest_points_4 * nDim ] ) / denom_z;

   gra20 = ( flowmap[ closest_points_1 * nDim + 1] - flowmap[ closest_points_0 * nDim + 1] ) / denom_x;
   gra21 = ( flowmap[ closest_points_3 * nDim + 1] - flowmap[ closest_points_2 * nDim + 1] ) / denom_y;
      gra22 = ( flowmap[ closest_points_5 * nDim + 1] - flowmap[ closest_points_4 * nDim + 1] ) / denom_z;

   gra30 = ( flowmap[ closest_points_1 * nDim + 2] - flowmap[ closest_points_0 * nDim + 2] ) / denom_x;
   gra31 = ( flowmap[ closest_points_3 * nDim + 2] - flowmap[ closest_points_2 * nDim + 2] ) / denom_y;
      gra32 = ( flowmap[ closest_points_5 * nDim + 2] - flowmap[ closest_points_4 * nDim + 2] ) / denom_z;
  }
  else
  {
   gra10 = 1;
   gra11 = 1;
   gra12 = 1;
   gra20 = 1;
   gra21 = 1;
   gra22 = 1;
   gra30 = 1;
   gra31 = 1;
   gra32 = 1;
  }


  ftle_matrix[0] = gra10 * gra10 + gra20 * gra20 + gra30 * gra30;
  ftle_matrix[1] = gra10 * gra11 + gra20 * gra21 + gra30 * gra31;
  ftle_matrix[2] = gra10 * gra12 + gra20 * gra22 + gra30 * gra32;
  ftle_matrix[3] = ftle_matrix[1];
  ftle_matrix[4] = gra11 * gra11 + gra21 * gra21 + gra31 * gra31;
  ftle_matrix[5] = gra11 * gra12 + gra11 * gra22 + gra31 * gra32;
  ftle_matrix[6] = ftle_matrix[2];
  ftle_matrix[7] = ftle_matrix[5];
  ftle_matrix[8] = gra12 * gra12 + gra22 * gra22 + gra32 * gra32;


  gra10 = ftle_matrix[0];
  gra11 = ftle_matrix[1];
  gra12 = ftle_matrix[2];
  gra20 = ftle_matrix[3];
  gra21 = ftle_matrix[4];
  gra22 = ftle_matrix[5];
  gra30 = ftle_matrix[6];
  gra31 = ftle_matrix[7];
  gra32 = ftle_matrix[8];


  double A10 = gra10 * gra10 + gra11 * gra11 + gra12 * gra12;
  double A11 = gra10 * gra20 + gra11 * gra21 + gra12 * gra22;
  double A12 = gra10 * gra30 + gra11 * gra31 + gra12 * gra32;
  double A20 = gra20 * gra10 + gra21 * gra11 + gra22 * gra12;
  double A21 = gra20 * gra20 + gra21 * gra21 + gra22 * gra22;
  double A22 = gra20 * gra30 + gra21 * gra31 + gra22 * gra32;
  double A30 = gra30 * gra10 + gra31 * gra11 + gra32 * gra12;
  double A31 = gra30 * gra20 + gra31 * gra21 + gra32 * gra22;
  double A32 = gra30 * gra30 + gra31 * gra31 + gra32 * gra32;


  double a = -1;
  double b = A10 + A21 + A32;
  double c = A12 * A30 + A22 * A31 + A11 * A20 - A10 * A21 - A10 * A32 - A21 * A32;
  double d = A10 * A21 * A32 + A11 * A22 * A30 + A12 * A20 * A31 - A10 * A22 * A31 - A11 * A20 * A32 - A12 * A21 * A30;
  double x1, x2, x3;
  double A = b*b - 3*a*c;
  double B = b*c - 9*a*d;
  double C = c*c - 3*b*d;
  double del = B*B - 4*A*C;
  if ( A == B && A == 0 )
  {
   x1 = x2 = x3 = -b/(3*a);
  }
  else if(del==0)
  {
   x1 = -b/a+B/A;
   x2 = x3 = (-B/A)/2;
  }
  else if(del<0)
  {
   double sqA = sycl::sqrt(A);
   double sq3 = sycl::sqrt(3.0);
   double T = (2*A*b-3*a*B) / (2*A*sqA);
   double _xt = sycl::acos(T);
   double xt = _xt/3;
   x1 = (-b-2*sqA*sycl::cos(xt)) / (3*a);
   x2 = (-b+sqA*(sycl::cos(xt)+sq3*sycl::sin(xt)))/(3*a);
   x3 = (-b+sqA*(sycl::cos(xt)-sq3*sycl::sin(xt)))/(3*a);
  }
  double max = x1;
  if (x2 > max ) max = x2;
  if (x3 > max ) max = x3;

  max = sycl::sqrt(max);
  max = sycl::log (max);

  logSqrt[i.get_global_id(0)] = max / T;
     }
 });
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


 if (argc < 8)
 {
  printf("USAGE: %s <nDim> <coords_file> <faces_file> <flowmap_file> <t_eval> <print2file> <nDevices>\n", argv[0]);
  printf("\tnDim:    dimensions of the space (2D/3D)\n");
  printf("\tcoords_file:   file where mesh coordinates are stored.\n");
  printf("\tfaces_file:    file where mesh faces are stored.\n");
  printf("\tflowmap_file:  file where flowmap values are stored.\n");
  printf("\tt_eval:        time when compute ftle is desired.\n");
  printf("\tprint to file? (0-NO, 1-YES)\n");
  printf("\tnDevices:       number of GPUs\n");
  return 1;
 }

 double t_eval = atof(argv[5]);
 int check_EOF;
 int nDevices = atoi(argv[7]);
 char buffer[255];
 int nDim, nVertsPerFace, nPoints, nFaces;
 FILE *file;
 double *coords;
 double *flowmap;
 int *faces;
 double *logSqrt;
 int *nFacesPerPoint;
 int *facesPerPoint;

 auto m_property_list =property_list{property::queue::enable_profiling(), property::queue::in_order()};
 auto devs = device::get_devices(info::device_type::gpu);
 queue queues[nDevices];
 for (int d=0; d< nDevices; d++){
   queues[d] = queue(devs[d], m_property_list);
  }

 nDim = atoi(argv[1]);
 if ( nDim == 2 ) nVertsPerFace = 3;
 else {
  if ( nDim == 3) nVertsPerFace = 4;
  else
  {
   printf("Wrong dimension provided (2 or 3 supported)\n");
   return 1;
  }
 }

 printf("\nReading input data\n\n");
 printf("\tReading mesh points coordinates...		");
 file = fopen( argv[2], "r" );
 check_EOF = fscanf(file, "%s", buffer);
 if ( check_EOF == (-1) )
 {
  fprintf( stderr, "Error: Unexpected EOF in read_coordinates\n" );
   exit(-1);
 }
 nPoints = atoi(buffer);
 fclose(file);
 coords = malloc_shared<double> (nPoints * nDim, queues[0]);
 read_coordinates(argv[2], nDim, nPoints, coords);
 printf("DONE\n");


 printf("\tReading mesh faces vertices...			");
 file = fopen( argv[3], "r" );
 check_EOF = fscanf(file, "%s", buffer);
 if ( check_EOF == (-1) )
 {
  fprintf( stderr, "Error: Unexpected EOF in read_faces\n" );
   exit(-1);
 }
 nFaces = atoi(buffer);
 faces = malloc_shared<int> (nFaces * nVertsPerFace, queues[0]);
 read_faces(argv[3], nDim, nVertsPerFace, nFaces, faces);
 printf("DONE\n");


 printf("\tReading mesh flowmap (x, y[, z])...	   ");
 flowmap = malloc_shared<double>(nPoints * nDim, queues[0]);
 read_flowmap ( argv[4], nDim, nPoints, flowmap );
 printf("DONE\n\n");
 printf("--------------------------------------------------------\n");

 nFacesPerPoint = malloc_shared<int>(nPoints, queues[0]);
 create_nFacesPerPoint_vector ( nDim, nPoints, nFaces, nVertsPerFace, faces, nFacesPerPoint );
 logSqrt= malloc_shared<double>(nPoints, queues[0]);
 facesPerPoint = malloc_shared<int>(nFacesPerPoint[ nPoints - 1 ], queues[0]);
 int v_points[nDevices];
 int offsets[nDevices];
 int offsets_faces[nDevices];
 event event_list[nDevices*2];
 int gap= ((nPoints / nDevices)/512)*512;
 for(int d=0; d < nDevices; d++){
  v_points[d] = (d == nDevices-1) ? nPoints - gap*d : gap;
  offsets[d] = gap*d;
 }
 for(int d=0; d < nDevices; d++){
  int inf = (d != 0) ? nFacesPerPoint[offsets[d]-1] : 0;
  int sup = (d != nDevices-1) ? nFacesPerPoint[offsets[d+1]-1] : nFacesPerPoint[nPoints-1];
  offsets_faces[d] = (d != 0) ? nFacesPerPoint[offsets[d]-1]: 0;
 }

 printf("\nComputing FTLE (SYCL USM SHARED)...");
 struct timeval global_timer_start;
 gettimeofday(&global_timer_start, __null);

 {
  for(int d=0; d < nDevices; d++){
   int* p_faces = facesPerPoint + offsets_faces[d];
   event_list[d] = create_facesPerPoint_vector(&queues[d], nDim, v_points[d], offsets[d], offsets_faces[d], nFaces, nVertsPerFace, faces, nFacesPerPoint, p_faces);
   double* p_logSqrt = logSqrt + offsets[d];
   if ( nDim == 2 )
    event_list[nDevices + d] = compute_gradient_2D ( &queues[d], v_points[d], offsets[d], offsets_faces[d], nVertsPerFace, coords, flowmap, faces, nFacesPerPoint,p_faces,p_logSqrt, t_eval);
     else
    event_list[nDevices + d] = compute_gradient_3D ( &queues[d], v_points[d], offsets[d], offsets_faces[d], nVertsPerFace, coords, flowmap, faces, nFacesPerPoint,p_faces, p_logSqrt, t_eval);

  }
  for(int d=0; d < nDevices; d++)
   queues[d].wait();
 }
 struct timeval global_timer_end;
 gettimeofday(&global_timer_end, __null);
 double time = (global_timer_end.tv_sec - global_timer_start.tv_sec) + (global_timer_end.tv_usec - global_timer_start.tv_usec)/1000000.0;
 printf("DONE\n\n");
 printf("--------------------------------------------------------\n");

 if ( atoi(argv[6]) )
 {
  printf("\nWriting result in output file...				  ");
   FILE *fp_w = fopen("usm_result.csv", "w");
  for ( int ii = 0; ii < nPoints; ii++ )
   fprintf(fp_w, "%f\n", logSqrt[ii]);
  fclose(fp_w);
  fp_w = fopen("usm_preproc.csv", "w");
  for ( int ii = 0; ii < nFacesPerPoint[nPoints-1]; ii++ )
   fprintf(fp_w, "%d\n", facesPerPoint[ii]);
  fclose(fp_w);
  printf("DONE\n\n");
  printf("--------------------------------------------------------\n");
  }

 printf("Execution times in miliseconds\n");
 printf("Device Num;  Preproc kernel; FTLE kernel\n");
 for(int d = 0; d < nDevices; d++){
  float preproc =  (event_list[d].get_profiling_info<info::event_profiling::command_end>() - event_list[d].get_profiling_info<info::event_profiling::command_start>()) / 1000000.0f;
  float ker =  (event_list[nDevices +d].get_profiling_info<info::event_profiling::command_end>() - event_list[nDevices + d].get_profiling_info<info::event_profiling::command_start>()) / 1000000.0f;
  printf("%d; %f; %f\n", d, preproc, ker);
 }
 printf("Global time: %f:\n", time);
 printf("--------------------------------------------------------\n");

 free(coords, queues[0]);
 free(faces, queues[0]);
 free(flowmap, queues[0]);
 free(logSqrt, queues[0]);
 free(facesPerPoint, queues[0]);
 free(nFacesPerPoint, queues[0]);

 return 0;
}
