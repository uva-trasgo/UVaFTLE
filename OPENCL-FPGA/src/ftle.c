#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>

#include "ftle.h"
#include "preprocess.h"

#include "opencl_helper.h"

int main(int argc, char *argv[]) {

    printf("--------------------------------------------------------\n");
    printf("|                        UVaFTLE                       |\n");
    printf("|                                                      |\n");
    printf("| Developers:                                          |\n");
    printf("|  - Rocío Carratalá-Sáez | rocio@infor.uva.es         |\n");
    printf("|  - Yuri Torres          | yuri.torres@infor.uva.es   |\n");
    printf("|  - Sergio López-Huguet  | serlohu@upv.es             |\n");
    printf("|  - Manuel de Castro     | manuel@infor.uva.es        |\n");
    printf("--------------------------------------------------------\n");
    fflush(stdout);

    // Check usage
    if (argc != 7)
    {
        printf("USAGE: ./executable <nDim> <coords_file> <faces_file> <flowmap_file> <t_eval>\n");
        printf("\texecutable:    compute_ftle.\n");
        printf("\tnDim:          dimensions of the space (2D/3D).\n");
        printf("\tcoords_file:   file where mesh coordinates are stored.\n");
        printf("\tfaces_file:    file where mesh faces are stored.\n");
        printf("\tflowmap_file:  file where flowmap values are stored.\n");
        printf("\tt_eval:        time when compute ftle is desired.\n");
        printf("\tprint2file:    store result in output file.\n");
	    exit(EXIT_FAILURE);
    }

    int check_EOF;
    int nDevices ;
    char buffer[255];
    int nDim, nVertsPerFace, nPoints, nFaces;
    FILE *file;

    double  t_eval =  atof(argv[5]), maxTime;
    double  times[10];
    int     numThreads = 0;
    double *coords;
    double *flowmap;
    int    *faces;
    double *logSqrt;

    int    *nFacesPerPoint;
    int    *facesPerPoint;

    int m, lda;

    /* Initialize mesh original information */
    nDim = atoi(argv[1]);
    m = nDim; lda = nDim;
    if (nDim == 2) nVertsPerFace = 3; // 2D: faces are triangles
    else if (nDim == 3) nVertsPerFace = 4; // 3D: faces (volumes) are tetrahedron
    else
    {
        printf("Wrong dimension provided (2 or 3 supported)\n");
        exit(EXIT_FAILURE);
    }

    /* Read coordinates, faces and flowmap from Python-generated files and generate corresponding vectors */
    /* Read coordinates information */
    printf("\nReading input data\n\n"); 
    printf("\tReading mesh points coordinates...        "); 
    fflush(stdout);

    file = fopen(argv[2], "r");
    check_EOF = fscanf(file, "%s", buffer);
    if (check_EOF == EOF)
    {
        fprintf(stderr, "Error: Unexpected EOF in read_coordinates\n"); 
        fflush(stdout);
        exit(EXIT_FAILURE);
    }
    nPoints = atoi(buffer);
    fclose(file);
    coords = (double *) malloc (sizeof(double) * nPoints * nDim);
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
        exit(EXIT_FAILURE);
    }
    nFaces = atoi(buffer);
    faces = (int *) malloc (sizeof(int) * nFaces * nVertsPerFace);
    read_faces(argv[3], nDim, nVertsPerFace, nFaces, faces); 
    printf("DONE\n"); 
    fflush(stdout);

    /* Read flowmap information */
    printf("\tReading mesh flowmap (x, y[, z])...       "); 
    fflush(stdout);
    flowmap = (double*) malloc(sizeof(double) * nPoints * nDim); 
    read_flowmap (argv[4], nDim, nPoints, flowmap);
    printf("DONE\n\n"); 
    printf("--------------------------------------------------------\n"); 
    fflush(stdout);

    /* Allocate additional memory at the CPU */    
    logSqrt        = (double *) malloc(sizeof(double) * nPoints);       
    nFacesPerPoint = (int *) malloc(sizeof(int) * nPoints); /* REMARK: nFacesPerPoint accumulates previous nFacesPerPoint */
    logSqrt        = (double *) malloc(sizeof(double) * nPoints);    



    /* Create OpenCL context and queue for FPGA and retrieve kernels */
    cl_device_id fpga_id;
    cl_context ctx = cl_context_and_device_for_platform(
#ifdef EMULATION
        "Intel(R) FPGA Emulation Platform for OpenCL(TM)",
#else // !EMULATION
        "Intel(R) FPGA SDK for OpenCL(TM)",
#endif // EMULATION
        &fpga_id);

        cl_queue queue = clCreateCommandQueue(ctx, device_id, 0, &err);
        OPENCL_CHECK_ERROR( err );

    const char *kernel_names[3] = { "create_facesPerPoint_vector_FPGA", "fpga_compute_gradient_2D", "fpga_compute_gradient_3D" };
    cl_kernel *fpga_kernels = cl_kernels_from_names("ftle_kernels.aocx", ctx, fpga_id, 3, kernel_names);
    cl_kernel create_facesPerPoint_vector_FPGA = fpga_kernels[0];
    cl_kernel fpga_compute_gradient_2D = fpga_kernels[1];
    cl_kernel fpga_compute_gradient_3D = fpga_kernels[2];
    free(fpga_kernels);

    /* Assign faces to vertices and generate nFacesPerPoint and facesPerPoint FPGA vectors */
    create_nFacesPerPoint_vector(nDim, nPoints, nFaces, nVertsPerFace, faces, nFacesPerPoint);
    facesPerPoint = (int *) malloc(sizeof(int) * nFacesPerPoint[ nPoints - 1 ]);

    cl_event kernel_event, read_event;
    cl_mem d2_facesPerPoint, d2_faces, d2_nFacesPerPoint;

    d2_facesPerPoint = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(int) * nFacesPerPoint[nPoints - 1], NULL, &err);
    OPENCL_CHECK_ERROR( err );
    d2_faces = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * nFaces * nVertsPerFace, faces, &err);
    OPENCL_CHECK_ERROR( err );
    d2_nFacesPerPoint = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * nPoints, nFacesPerPoint, &err);
    OPENCL_CHECK_ERROR( err );

    OPENCL_CHECK( clSetKernelArg(create_facesPerPoint_vector_FPGA, 0, sizeof(cl_int), &nDim) );
    OPENCL_CHECK( clSetKernelArg(create_facesPerPoint_vector_FPGA, 1, sizeof(cl_int), &nPoints) );
    OPENCL_CHECK( clSetKernelArg(create_facesPerPoint_vector_FPGA, 2, sizeof(cl_int), &nFaces) );
    OPENCL_CHECK( clSetKernelArg(create_facesPerPoint_vector_FPGA, 3, sizeof(cl_int), &nVertsPerFace) );
    OPENCL_CHECK( clSetKernelArg(create_facesPerPoint_vector_FPGA, 4, sizeof(cl_mem), &d2_faces) );
    OPENCL_CHECK( clSetKernelArg(create_facesPerPoint_vector_FPGA, 5, sizeof(cl_mem), &d2_nFacesPerPoint) );
    OPENCL_CHECK( clSetKernelArg(create_facesPerPoint_vector_FPGA, 6, sizeof(cl_mem), &d2_facesPerPoint) );
    OPENCL_CHECK( clEnqueueTask(queue, create_facesPerPoint_vector_FPGA, 0, NULL, &kernel_event) );
    OPENCL_CHECK( clEnqueueReadBuffer(queue, d2_facesPerPoint, CL_TRUE, 0, sizeof(int) * nFacesPerPoint[nPoints - 1], &facesPerPoint, 1, &kernel_event, NULL) );

    printf("\nComputing FTLE...                                 ");
    fflush(stdout);

    cl_mem d_logSqrt;
    cl_mem d_gra1, d_gra2, d_gra3;
    cl_mem d_coords, d_flowmap, d_res;
    cl_mem d_faces, d_nFacesPerPoint, d_facesPerPoint;
    cudaSetDevice(omp_get_thread_num());

    /* Allocate and copy memory for read data at the FPGA (coords, faces, flowmap) */
    d_coords = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * nPoints * nDim, coords, &err);
    OPENCL_CHECK_ERROR( err );
    d_faces = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * nFaces * nVertsPerFace, faces, &err);
    OPENCL_CHECK_ERROR( err );
    d_flowmap = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * nPoints * nDim, flowmap, &err);
    OPENCL_CHECK_ERROR( err );

    /* Allocate additional memory at the GPU */
    d_nFacesPerPoint = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , sizeof(int) * nPoints, nFacesPerPoint, &err);
    OPENCL_CHECK_ERROR( err );
    d_logSqrt = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(double) * nPoints, NULL, &err);
    OPENCL_CHECK_ERROR( err );
    d_res = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(double) * nPoints * nDim * nDim, NULL, &err);
    OPENCL_CHECK_ERROR( err );
    d_gra1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(double) * nPoints * nDim, NULL, &err);
    OPENCL_CHECK_ERROR( err );
    d_gra2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(double) * nPoints * nDim, NULL, &err);
    OPENCL_CHECK_ERROR( err );
    if (nDim == 3)
    {
        d_flowmap = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(double) * nPoints * nDim, NULL, &err);
        OPENCL_CHECK_ERROR( err );
    }
    d_facesPerPoint = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * nFacesPerPoint[nPoints - 1], facesPerPoint, &err);
    OPENCL_CHECK_ERROR( err );

    struct timeval start;
    struct timeval end;                 
    double time;

    /* Solve FTLE */
    OPENCL_CHECK( clFinish(queue) );
    gettimeofday(&start, NULL);

    /* STEP 1: compute gradient, tensors and ATxA based on neighbors flowmap values */

    if (nDim == 2)
        OPENCL_CHECK( clSetKernelArg(fpga_compute_gradient_2D, 0, sizeof(cl_int), &nPoints) );
        OPENCL_CHECK( clSetKernelArg(fpga_compute_gradient_2D, 1, sizeof(cl_int), &nVertsPerFace) );
        OPENCL_CHECK( clSetKernelArg(fpga_compute_gradient_2D, 2, sizeof(cl_mem), &d_coords) );
        OPENCL_CHECK( clSetKernelArg(fpga_compute_gradient_2D, 3, sizeof(cl_mem), &d_flowmap) );
        OPENCL_CHECK( clSetKernelArg(fpga_compute_gradient_2D, 4, sizeof(cl_mem), &d_faces) );
        OPENCL_CHECK( clSetKernelArg(fpga_compute_gradient_2D, 5, sizeof(cl_mem), &d_nFacesPerPoint) );
        OPENCL_CHECK( clSetKernelArg(fpga_compute_gradient_2D, 6, sizeof(cl_mem), &d_facesPerPoint) );
        OPENCL_CHECK( clSetKernelArg(fpga_compute_gradient_2D, 7, sizeof(cl_mem), &d_logSqrt) );
        OPENCL_CHECK( clSetKernelArg(fpga_compute_gradient_2D, 8, sizeof(cl_double), &t_eval) );
        OPENCL_CHECK( clEnqueueTask(queue, fpga_compute_gradient_2D, 0, NULL, &kernel_event) );
        OPENCL_CHECK( clEnqueueReadBuffer(queue, d2_facesPerPoint, CL_TRUE, 0, sizeof(int) * nFacesPerPoint[nPoints - 1], &facesPerPoint, 1, &kernel_event, NULL) );
    else
        OPENCL_CHECK( clSetKernelArg(fpga_compute_gradient_3D, 0, sizeof(cl_int), &nPoints) );
        OPENCL_CHECK( clSetKernelArg(fpga_compute_gradient_3D, 1, sizeof(cl_int), &nVertsPerFace) );
        OPENCL_CHECK( clSetKernelArg(fpga_compute_gradient_3D, 2, sizeof(cl_mem), &d_coords) );
        OPENCL_CHECK( clSetKernelArg(fpga_compute_gradient_3D, 3, sizeof(cl_mem), &d_flowmap) );
        OPENCL_CHECK( clSetKernelArg(fpga_compute_gradient_3D, 4, sizeof(cl_mem), &d_faces) );
        OPENCL_CHECK( clSetKernelArg(fpga_compute_gradient_3D, 5, sizeof(cl_mem), &d_nFacesPerPoint) );
        OPENCL_CHECK( clSetKernelArg(fpga_compute_gradient_3D, 6, sizeof(cl_mem), &d_facesPerPoint) );
        OPENCL_CHECK( clSetKernelArg(fpga_compute_gradient_3D, 7, sizeof(cl_mem), &d_logSqrt) );
        OPENCL_CHECK( clSetKernelArg(fpga_compute_gradient_3D, 8, sizeof(cl_double), &t_eval) );
        OPENCL_CHECK( clEnqueueTask(queue, fpga_compute_gradient_3D, 0, NULL, &kernel_event) );

    cudaMemcpyAsync (&logSqrt[localStride],  &d_logSqrt[localStride], sizeof(double) * (nPoints / omp_get_num_threads()), cudaMemcpyDeviceToHost,streams[0]);


    OPENCL_CHECK( clFinish(queue) );
    /* Time */
    gettimeofday(&end, NULL);

    /* Show time */   
    time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1000000.0;

    /* Free memory */
    OPENCL_CHECK( clReleaseMemObject(d_coords) );
    OPENCL_CHECK( clReleaseMemObject(d_flowmap) );
    OPENCL_CHECK( clReleaseMemObject(d_faces) );
    OPENCL_CHECK( clReleaseMemObject(d_nFacesPerPoint) );
    OPENCL_CHECK( clReleaseMemObject(d_facesPerPoint) );
    OPENCL_CHECK( clReleaseMemObject(d_logSqrt) );
    OPENCL_CHECK( clReleaseMemObject(d_gra1) );
    OPENCL_CHECK( clReleaseMemObject(d_gra2) );
    OPENCL_CHECK( clReleaseMemObject(d_gra3) );
    OPENCL_CHECK( clReleaseMemObject(d_res) );

    OPENCL_CHECK( clReleaseKernel(create_facesPerPoint_vector_FPGA) );
    OPENCL_CHECK( clReleaseKernel(fpga_compute_gradient_2D) );
    OPENCL_CHECK( clReleaseKernel(fpga_compute_gradient_3D) );
    OPENCL_CHECK( clReleaseCommandQueue(queue) );
    OPENCL_CHECK( clReleaseContext(context) );

    printf("DONE\n\n");
    printf("--------------------------------------------------------\n");
    fflush(stdout);

    /* Write result in output file (if desired) */
    if (atoi(argv[6]))
    {
        printf("\nWriting result in output file...                  ");
        fflush(stdout);
        FILE *fp_w = fopen("result_FTLE.csv", "w");
        for (int ii = 0; ii < nPoints ; ii++)
            fprintf(fp_w, "%f\n", logSqrt[ii]);
        fclose(fp_w);
        printf("DONE\n\n");
        printf("--------------------------------------------------------\n");
        fflush(stdout);
    }

    /* Show execution time */
    maxTime = times[0];
    for (int i = 1; i < numThreads; i++){
        if (times[i] > maxTime)
            maxTime = times[i];
    }
    printf("\nExecution time (seconds): %f\n\n", maxTime);
    printf("--------------------------------------------------------\n");
    fflush(stdout);
    
    /* Free memory */
    free(coords);
    free(faces);
    free(flowmap);

    free(logSqrt);

    free(nFacesPerPoint);
    free(facesPerPoint);

    return 0;
}
