#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <CL/sycl.hpp>
using namespace cl::sycl;

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

cl::sycl::event create_facesPerPoint_vector (queue* q, int nDim, int nPoints, int offset, int faces_offset, int nFaces, int nVertsPerFace, cl::sycl::buffer<int, 1> *b_faces, cl::sycl::buffer<int, 1> *b_nFacesPerPoint, cl::sycl::buffer<int, 1> *b_facesPerPoint)
{
return q->submit([&](handler &h){
 auto faces = b_faces->get_access<access::mode::read>(h);
 auto nFacesPerPoint = b_nFacesPerPoint->get_access<access::mode::read>(h);
 auto facesPerPoint = b_facesPerPoint->get_access<access::mode::discard_write>(h);
 int size = (nPoints% 512) ? (nPoints/512 +1)*512: nPoints;
 h.parallel_for<class preprocess> (nd_range<1>(range<1>{static_cast<size_t>(size)},range<1>{static_cast<size_t>(512)}), [=](nd_item<1> i){
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
});
}
::event compute_gradient_2D (queue* q, int nPoints, int offset, int faces_offset, int nVertsPerFace, ::buffer<double, 1> *b_coords, ::buffer<double, 1> *b_flowmap, ::buffer<int, 1> *b_faces, ::buffer<int, 1> *b_nFacesPerPoint, ::buffer<int, 1> *b_facesPerPoint, ::buffer<double, 1> *b_log_sqrt, double T )
{
return q->submit([&](handler &h){
 auto coords = b_coords->get_access<access::mode::read>(h);
 auto flowmap = b_flowmap->get_access<access::mode::read>(h);
 auto faces = b_faces->get_access<access::mode::read>(h);
 auto nFacesPerPoint = b_nFacesPerPoint->get_access<access::mode::read>(h);
 auto facesPerPoint = b_facesPerPoint->get_access<access::mode::read>(h);
 auto d_logSqrt = b_log_sqrt->get_access<access::mode::discard_write>(h);

 int size = (nPoints% 512) ? (nPoints/512 +1)*512: nPoints;
 h.parallel_for<class ftle2D> (nd_range<1>(range<1>{static_cast<size_t>(size)},range<1>{static_cast<size_t>(512)}), [=](nd_item<1> i){
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
  double sq = cl::sycl::sqrt(A21 * A21 + A10 * A10 - 2 * (A10 * A21) + 4 * (A11 * A20));
  d_W_ei[0] = (A21 + A10 + sq) / 2;
  d_W_ei[1] = (A21 + A10 - sq) / 2;


  double max = d_W_ei[0];
  if (d_W_ei[1] > max ) max = d_W_ei[1];
  max = cl::sycl::sqrt(max);
  max = cl::sycl::log (max);

  d_logSqrt[i.get_global_id(0)] = max / T;
     }
 });
});
}

::event compute_gradient_3D (queue* q, int nPoints, int offset, int faces_offset, int nVertsPerFace, ::buffer<double, 1> *b_coords, ::buffer<double, 1> *b_flowmap, ::buffer<int, 1> *b_faces, ::buffer<int, 1> *b_nFacesPerPoint, ::buffer<int, 1> *b_facesPerPoint, ::buffer<double, 1> *b_log_sqrt, double T )
{
return q->submit([&](handler &h){
 auto coords = b_coords->get_access<access::mode::read, access::target::constant_buffer>(h);
 auto flowmap = b_flowmap->get_access<access::mode::read, access::target::constant_buffer>(h);
 auto faces = b_faces->get_access<access::mode::read, access::target::constant_buffer>(h);
 auto nFacesPerPoint = b_nFacesPerPoint->get_access<access::mode::read, access::target::constant_buffer>(h);
 auto facesPerPoint = b_facesPerPoint->get_access<access::mode::read, access::target::constant_buffer>(h);
 auto d_logSqrt = b_log_sqrt->get_access<access::mode::discard_write, access::target::global_buffer>(h);

 int size = (nPoints% 512) ? (nPoints/512 +1)*512: nPoints;
 h.parallel_for<class ftle3D> (nd_range<1>(range<1>{static_cast<size_t>(size)},range<1>{static_cast<size_t>(512)}), [=](nd_item<1> i){
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
   double sqA = cl::sycl::sqrt(A);
   double sq3 = cl::sycl::sqrt(3.0);
   double T = (2*A*b-3*a*B) / (2*A*sqA);
   double _xt = cl::sycl::acos(T);
   double xt = _xt/3;
   x1 = (-b-2*sqA*cl::sycl::cos(xt)) / (3*a);
   x2 = (-b+sqA*(cl::sycl::cos(xt)+sq3*cl::sycl::sin(xt)))/(3*a);
   x3 = (-b+sqA*(cl::sycl::cos(xt)-sq3*cl::sycl::sin(xt)))/(3*a);
  }
  double max = x1;
  if (x2 > max ) max = x2;
  if (x3 > max ) max = x3;

  max = cl::sycl::sqrt(max);
  max = cl::sycl::log (max);
  d_logSqrt[i.get_global_id(0)] = max / T;
     }
 });
});
}
float getKernelExecutionTime(::event event){
 auto start_time = event.get_profiling_info<::info::event_profiling::command_start>();
  auto end_time = event.get_profiling_info<cl::sycl::info::event_profiling::command_end>();
  return (end_time - start_time) / 1000000.0f;
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


 if (argc != 8)
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

 auto property_list =::property_list{::property::queue::enable_profiling()};
 auto devs = device::get_devices(info::device_type::gpu);
 queue queues[nDevices];
 for (int d=0; d< nDevices; d++){
   queues[d] = queue(devs[d], property_list);
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
 fflush(stdout);
 printf("\tReading mesh points coordinates...		");
 fflush(stdout);
 file = fopen( argv[2], "r" );
 check_EOF = fscanf(file, "%s", buffer);
 if ( check_EOF == (-1) )
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


 printf("\tReading mesh faces vertices...			");
 fflush(stdout);
 file = fopen( argv[3], "r" );
 check_EOF = fscanf(file, "%s", buffer);
 if ( check_EOF == (-1) )
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


 printf("\tReading mesh flowmap (x, y[, z])...	   ");
 fflush(stdout);
 flowmap = (double*) malloc( sizeof(double) * nPoints * nDim );
 read_flowmap ( argv[4], nDim, nPoints, flowmap );
 printf("DONE\n\n");
 printf("--------------------------------------------------------\n");
 fflush(stdout);


 nFacesPerPoint = (int *) malloc( sizeof(int) * nPoints );

 create_nFacesPerPoint_vector ( nDim, nPoints, nFaces, nVertsPerFace, faces, nFacesPerPoint );
 logSqrt= (double*) malloc( sizeof(double) * nPoints);
 facesPerPoint = (int *) malloc( sizeof(int) * nFacesPerPoint[ nPoints - 1 ] );
 int v_points[8] = {1,1,1,1,1,1,1,1};
 int offsets[8] = {0,0,0,0,0,0,0,0};
 int v_points_faces[8] = {1,1,1,1,1,1,1,1};
 int offsets_faces[8] = {0,0,0,0,0,0,0,0};

 ::event event_list[nDevices*2];
 int gap= ((nPoints / nDevices)/512)*512;
 for(int d=0; d < nDevices; d++){
  v_points[d] = (d == nDevices-1) ? nPoints - gap*d : gap;
  offsets[d] = gap*d;
 }
 for(int d=0; d < nDevices; d++){
  int inf = (d != 0) ? nFacesPerPoint[offsets[d]-1] : 0;
  int sup = (d != nDevices-1) ? nFacesPerPoint[offsets[d+1]-1] : nFacesPerPoint[nPoints-1];
  v_points_faces[d] = sup - inf;
  offsets_faces[d] = (d != 0) ? nFacesPerPoint[offsets[d]-1]: 0;
 }

 printf("\nComputing FTLE (SYCL)...");
 struct timeval global_timer_start;
 gettimeofday(&global_timer_start, __null);

 {
  ::buffer<double, 1> b_coords(coords, range<1>{static_cast<size_t>(nPoints * nDim)});
  ::buffer<int, 1> b_faces(faces, range<1>{static_cast<size_t>(nFaces * nVertsPerFace)});
  ::buffer<double, 1> b_flowmap(flowmap, range<1>{static_cast<size_t>(nPoints*nDim)});
  ::buffer<int, 1> b_nFacesPerPoint(nFacesPerPoint, range<1>{static_cast<size_t>(nPoints)});
  ::buffer<int, 1> b_faces0(facesPerPoint + offsets_faces[0], range<1>{static_cast<size_t>(v_points_faces[0])});
  ::buffer<int, 1> b_faces1(facesPerPoint + offsets_faces[1], range<1>{static_cast<size_t>(v_points_faces[1])});
  ::buffer<int, 1> b_faces2(facesPerPoint + offsets_faces[2], range<1>{static_cast<size_t>(v_points_faces[2])});
  ::buffer<int, 1> b_faces3(facesPerPoint + offsets_faces[3], range<1>{static_cast<size_t>(v_points_faces[3])});
  ::buffer<int, 1> b_faces4(facesPerPoint + offsets_faces[4], range<1>{static_cast<size_t>(v_points_faces[4])});
  ::buffer<int, 1> b_faces5(facesPerPoint + offsets_faces[5], range<1>{static_cast<size_t>(v_points_faces[5])});
  ::buffer<int, 1> b_faces6(facesPerPoint + offsets_faces[6], range<1>{static_cast<size_t>(v_points_faces[6])});
  ::buffer<int, 1> b_faces7(facesPerPoint + offsets_faces[7], range<1>{static_cast<size_t>(v_points_faces[7])});
  ::buffer<double, 1> b_logSqrt0(logSqrt + offsets[0], range<1>{static_cast<size_t>(v_points[0])}); 
  ::buffer<double, 1> b_logSqrt1(logSqrt + offsets[1], range<1>{static_cast<size_t>(v_points[1])});
  ::buffer<double, 1> b_logSqrt2(logSqrt + offsets[2], range<1>{static_cast<size_t>(v_points[2])});
  ::buffer<double, 1> b_logSqrt3(logSqrt + offsets[3], range<1>{static_cast<size_t>(v_points[3])});
  ::buffer<double, 1> b_logSqrt4(logSqrt + offsets[4], range<1>{static_cast<size_t>(v_points[4])}); 
  ::buffer<double, 1> b_logSqrt5(logSqrt + offsets[5], range<1>{static_cast<size_t>(v_points[5])});
  ::buffer<double, 1> b_logSqrt6(logSqrt + offsets[6], range<1>{static_cast<size_t>(v_points[6])});
  ::buffer<double, 1> b_logSqrt7(logSqrt + offsets[7], range<1>{static_cast<size_t>(v_points[7])});



  for(int d=0; d < nDevices; d++){
  ::buffer<int, 1> *used_faces = (d==0 ? &b_faces0 : (d==1 ? &b_faces1 : (d==2 ? &b_faces2 : (d==3 ? &b_faces3 : ((d==4 ? &b_faces4 : ((d==5 ? &b_faces5 : ((d==6 ? &b_faces6 : &b_faces7)))))))))); 
  ::buffer<double, 1> *used_sqrt = (d==0 ? &b_logSqrt0 : (d==1 ? &b_logSqrt1 : (d==2 ? &b_logSqrt2 : (d==3 ? &b_logSqrt3 : ((d==4 ? &b_logSqrt4 : ((d==5 ? &b_logSqrt5 : ((d==6 ? &b_logSqrt6 : &b_logSqrt7)))))))))); 
   event_list[d] = create_facesPerPoint_vector(&queues[d], nDim, v_points[d], offsets[d], offsets_faces[d], nFaces, nVertsPerFace, &b_faces, &b_nFacesPerPoint,used_faces);

   if ( nDim == 2 )
    event_list[nDevices + d] = compute_gradient_2D ( &queues[d], v_points[d], offsets[d], offsets_faces[d], nVertsPerFace, &b_coords, &b_flowmap, &b_faces, &b_nFacesPerPoint, used_faces, used_sqrt, t_eval);
     else
    event_list[nDevices + d] = compute_gradient_3D ( &queues[d], v_points[d], offsets[d], offsets_faces[d], nVertsPerFace, &b_coords, &b_flowmap, &b_faces, &b_nFacesPerPoint, used_faces, used_sqrt, t_eval);

  }
 }
 struct timeval global_timer_end;
 gettimeofday(&global_timer_end, __null);
 double time = (global_timer_end.tv_sec - global_timer_start.tv_sec) + (global_timer_end.tv_usec - global_timer_start.tv_usec)/1000000.0;
 printf("DONE\n\n");
 printf("--------------------------------------------------------\n");
 fflush(stdout);

 if ( atoi(argv[6]) )
 {
  printf("\nWriting result in output file...				  ");
  fflush(stdout);
  FILE *fp_w = fopen("sycl_result.csv", "w");
  for ( int ii = 0; ii < nPoints; ii++ )
   fprintf(fp_w, "%f\n", logSqrt[ii]);
  fclose(fp_w);
  fp_w = fopen("sycl_preproc.csv", "w");
                for ( int ii = 0; ii < nFacesPerPoint[nPoints-1]; ii++ )
                        fprintf(fp_w, "%d\n", facesPerPoint[ii]);
                fclose(fp_w);
  printf("DONE\n\n");
  printf("--------------------------------------------------------\n");
  fflush(stdout);
 }


 printf("Execution times in miliseconds\n");
 printf("Device Num;  Preproc kernel; FTLE kernel\n");
 for(int d = 0; d < nDevices; d++){
  printf("%d; %f; %f\n", d, getKernelExecutionTime(event_list[d]), getKernelExecutionTime(event_list[nDevices + d]));
 }
 printf("Global time: %f:\n", time);
 printf("--------------------------------------------------------\n");
 fflush(stdout);


 free(coords);
 free(faces);
 free(flowmap);
 free(logSqrt);
 free(facesPerPoint);
 free(nFacesPerPoint);

 return 0;
}
