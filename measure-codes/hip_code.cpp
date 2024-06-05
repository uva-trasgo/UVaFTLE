#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <omp.h>
#include <hip/hip_runtime.h>

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

__global__ void create_facesPerPoint_vector(int nDim, int nPoints, int offset, int faces_offset, int nFaces, int nVertsPerFace, int *faces, int *nFacesPerPoint, int *facesPerPoint )
{
 int gpu_id = blockIdx.x*blockDim.x + threadIdx.x;
 if (gpu_id < nPoints){
  int th_id = gpu_id + offset;
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
 }
}

__global__ void gpu_compute_gradient_2D(int nPoints, int offset, int faces_offset, int nVertsPerFace, double *coords, double *flowmap, int *faces, int *nFacesPerPoint, int *facesPerPoint, double *d_logSqrt, double T)
{
 int gpu_id = blockIdx.x*blockDim.x + threadIdx.x;
 if (gpu_id < nPoints){
  int th_id = gpu_id + offset;
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
  double sq = sqrt(A21 * A21 + A10 * A10 - 2 * (A10 * A21) + 4 * (A11 * A20));
  d_W_ei[0] = (A21 + A10 + sq) / 2;
  d_W_ei[1] = (A21 + A10 - sq) / 2;


  double max = d_W_ei[0];
  if (d_W_ei[1] > max ) max = d_W_ei[1];
  max = sqrt(max);
  max = log (max);
  d_logSqrt[gpu_id] = max / T;
 }
}

__global__ void gpu_compute_gradient_3D (int nPoints, int offset, int faces_offset, int nVertsPerFace, double *coords, double *flowmap, int *faces, int *nFacesPerPoint, int *facesPerPoint, double *d_logSqrt, double T)
{
 int gpu_id = blockIdx.x*blockDim.x + threadIdx.x;
 if (gpu_id < nPoints){
  int th_id = gpu_id + offset;
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
   double sqA = sqrt(A);
   double sq3 = sqrt(3.0);
   double T = (2*A*b-3*a*B) / (2*A*sqA);
   double _xt = acos(T);
   double xt = _xt/3;
   x1 = (-b-2*sqA*cos(xt)) / (3*a);
   x2 = (-b+sqA*(cos(xt)+sq3*sin(xt)))/(3*a);
   x3 = (-b+sqA*(cos(xt)-sq3*sin(xt)))/(3*a);
  }
  double max = x1;
  if (x2 > max ) max = x2;
  if (x3 > max ) max = x3;

  max = sqrt(max);
  max = log (max);
  d_logSqrt[gpu_id] = max / T;
 }
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
 int nDevices = atoi(argv[7]), maxDevices;
 char buffer[255];
 int nDim, nVertsPerFace, nPoints, nFaces;
 FILE *file;
 double *coords;
 double *flowmap;
 int *faces;
 double *logSqrt;
 int *nFacesPerPoint;
 int *facesPerPoint;


 hipGetDeviceCount(&maxDevices);
 float kernel_times[maxDevices];
 float preproc_times[maxDevices];


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

 hipHostMalloc( (void **) &logSqrt, sizeof(double) * nPoints);
 hipHostMalloc( (void **) &facesPerPoint, sizeof(int) * nFacesPerPoint[ nPoints - 1 ]);

 int v_points[maxDevices];
 int offsets[maxDevices];
 int v_points_faces[maxDevices];
 int offsets_faces[maxDevices];
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


 printf("\nComputing FTLE (HIP pinned)...");

 struct timeval global_timer_start;
 gettimeofday(&global_timer_start, __null);
#pragma omp parallel default(none) shared(stdout, logSqrt, nDim, nPoints, nFaces, nVertsPerFace, preproc_times, kernel_times, v_points, v_points_faces, offsets, offsets_faces, faces, coords, nFacesPerPoint, facesPerPoint, flowmap, t_eval)
 {
  int d = omp_get_thread_num();
  double *d_logSqrt;
  double *d_coords, *d_flowmap;
  int *d_faces, *d_nFacesPerPoint, *d_facesPerPoint;

  hipSetDevice(d);


  hipMalloc( &d_coords, sizeof(double) * nPoints * nDim );
  hipMalloc( &d_faces, sizeof(int) * nFaces * nVertsPerFace );
  hipMalloc( &d_flowmap, sizeof(double) * nPoints * nDim );


  hipMemcpy( d_coords, coords, sizeof(double) * nPoints * nDim, hipMemcpyHostToDevice );
  hipMemcpy( d_faces, faces, sizeof(int) * nFaces * nVertsPerFace, hipMemcpyHostToDevice );
  hipMemcpy( d_flowmap, flowmap, sizeof(double) * nPoints * nDim, hipMemcpyHostToDevice );


  hipMalloc( &d_nFacesPerPoint, sizeof(int) * nPoints);
  hipMalloc( &d_logSqrt, sizeof(double) * v_points[d]);


  hipMemcpy( d_nFacesPerPoint, nFacesPerPoint, sizeof(int) * nPoints, hipMemcpyHostToDevice );
  hipMalloc( &d_facesPerPoint, sizeof(int) * v_points_faces[d]);


  dim3 block(512);
  int numBlocks = (int) (ceil((double)v_points[d]/(double)block.x)+1);
  dim3 grid_numCoords(numBlocks+1);


  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  hipEventRecord(start, hipStreamDefault);

  hipLaunchKernelGGL(create_facesPerPoint_vector, grid_numCoords,block, 0, hipStreamDefault, nDim, v_points[d], offsets[d], offsets_faces[d], nFaces, nVertsPerFace, d_faces, d_nFacesPerPoint, d_facesPerPoint);

  hipEventRecord(stop, hipStreamDefault);
  hipEventSynchronize(stop);
  hipEventElapsedTime(preproc_times+ omp_get_thread_num() , start, stop);
  hipEventRecord(start, hipStreamDefault);

  if ( nDim == 2 )
   hipLaunchKernelGGL(gpu_compute_gradient_2D,grid_numCoords, block, 0, hipStreamDefault, v_points[d], offsets[d], offsets_faces[d], nVertsPerFace, d_coords, d_flowmap, d_faces, d_nFacesPerPoint, d_facesPerPoint, d_logSqrt, t_eval);
  else
   hipLaunchKernelGGL(gpu_compute_gradient_3D, grid_numCoords, block, 0, hipStreamDefault, v_points[d], offsets[d], offsets_faces[d], nVertsPerFace, d_coords, d_flowmap, d_faces, d_nFacesPerPoint, d_facesPerPoint, d_logSqrt, t_eval);

  hipEventRecord(stop, hipStreamDefault);
  hipEventSynchronize(stop);
  hipEventElapsedTime(kernel_times + omp_get_thread_num(), start, stop);





  hipMemcpyAsync (logSqrt + offsets[d], d_logSqrt, sizeof(double) * v_points[d], hipMemcpyDeviceToHost, hipStreamDefault);
  hipMemcpyAsync (facesPerPoint + offsets_faces[d], d_facesPerPoint, sizeof(int) * v_points_faces[d], hipMemcpyDeviceToHost, hipStreamDefault);
  hipDeviceSynchronize();


  fflush(stdout);


  hipFree(d_coords);
  hipFree(d_flowmap);
  hipFree(d_faces);
  hipFree(d_nFacesPerPoint);
  hipFree(d_facesPerPoint);
  hipFree(d_logSqrt);
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
  FILE *fp_w = fopen("rocm_result.csv", "w");
  for ( int ii = 0; ii < nPoints; ii++ )
   fprintf(fp_w, "%f\n", logSqrt[ii]);
  fclose(fp_w);
  fp_w = fopen("rocm_preproc.csv", "w");
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
  printf("%d; %f; %f\n", d, preproc_times[d], kernel_times[d]);
 }
 printf("Global time: %f:\n", time);
 printf("--------------------------------------------------------\n");
 fflush(stdout);


 free(coords);
 free(faces);
 free(flowmap);
 hipFree(logSqrt);
 hipFree(facesPerPoint);

 free(nFacesPerPoint);

 return 0;
}
