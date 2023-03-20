#include <string.h>

#include "opencl_helper.h"

typedef unsigned char byte;
typedef char int8;

cl_context cl_context_and_device_for_platform(const char *platform, cl_device_id *device_id)
{
	/* 1: OpenCL context initialization */
	cl_int err;

	cl_platform_id platform_id;

	cl_context context;

	/* 1.1: Retrieve available platforms */
	cl_uint n_platforms;
	OPENCL_CHECK( clGetPlatformIDs(0, NULL, &n_platforms) );
	cl_platform_id *p_platforms = (cl_platform_id *)malloc(n_platforms * sizeof(cl_platform_id));
	MALLOC_CHECK( p_platforms );
	OPENCL_CHECK( clGetPlatformIDs(n_platforms, p_platforms, NULL) );

	/* 1.2: Get platform names and choose accordingly */
	char *platform_name;

	int cl_platform_index = -1;
	for (int i = 0; i < (int)n_platforms; i++)
	{
		size_t platform_name_size;
		OPENCL_CHECK( clGetPlatformInfo(p_platforms[i], CL_PLATFORM_NAME, 0, NULL, &platform_name_size) );
		platform_name = (char *)malloc(platform_name_size * sizeof(char));
		MALLOC_CHECK( p_platforms );

		OPENCL_CHECK( clGetPlatformInfo(p_platforms[i], CL_PLATFORM_NAME, platform_name_size, platform_name, NULL) );
		printf("[OPENCL] Found platform %d: %s\n", i, platform_name);
		if (!strncmp(platform_name, platform, strlen(platform)))
		{
			cl_platform_index = i;
			break;
		}
		free(platform_name);
	}

	if (cl_platform_index < 0)
	{
		fprintf(stderr, "ERROR - Couldn't find a supported OpenCL platform\n");
		exit(EXIT_FAILURE);
	}
	platform_id = p_platforms[cl_platform_index];
	

	/* 1.3: Choose OpenCL device */
	cl_uint cl_devices;
	OPENCL_CHECK( clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &cl_devices) );
	if (cl_devices <= 0)
	{
		fprintf(stderr, "ERROR - Couldn't find a supported OpenCL device\n");
		exit(EXIT_FAILURE);
	}

	cl_device_id *p_devices = (cl_device_id *)malloc(cl_devices * sizeof(cl_device_id));
	MALLOC_CHECK( p_devices );
	OPENCL_CHECK( clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR, cl_devices, p_devices, NULL) );
	*device_id = p_devices[0];
	free(p_platforms);

	/* 1.4 Get and print device name */
	size_t device_name_size;
	OPENCL_CHECK( clGetDeviceInfo(*device_id, CL_DEVICE_NAME, 0, NULL, &device_name_size) );
	char *device_name = (char *)malloc(sizeof(char) * device_name_size);
	MALLOC_CHECK( device_name );
	OPENCL_CHECK( clGetDeviceInfo(*device_id, CL_DEVICE_NAME, device_name_size, device_name, NULL) );

	printf("\n ---------------- DEVICE INFORMATION ---------------- \n");
	printf(" PLATFORM: %s\n", platform_name);
	printf(" DEVICE: %s\n", device_name);
	printf("\n ---------------------------------------------------- \n");
	fflush(stdout);

	free(platform_name);
	free(device_name);

	/* 1.5: Create context */
	cl_context_properties context_properties[] = {
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)platform_id,
		0
	};
	context = clCreateContext(context_properties, 1, device_id, NULL, NULL, &err);
	OPENCL_CHECK_ERROR( err );

	return context;
}

cl_kernel* cl_kernels_from_names(const char *file_path, cl_context context, cl_device device_id, const int n_kernels, const char **entrypoints)
{
	cl_int err;

	cl_program program;
	cl_kernel *kernels;

	/* 1.6: Load kernel file */
	FILE *kernel_file;
	if (!(kernel_file = fopen(file_path, "rb")))
	{
		fprintf(stderr, "ERROR - Couldn't find the kernel file at %s\n", file_path);
		exit(EXIT_FAILURE);
	}

	fseek(kernel_file, 0, SEEK_END);
	size_t kernel_length = ftell(kernel_file);
	byte *kernel_bin = (byte *)malloc((kernel_length + 1) * sizeof(byte));
	MALLOC_CHECK( kernel_bin );
	rewind(kernel_file);
	if (!fread(kernel_bin, sizeof(byte), kernel_length, kernel_file))
	{
		fprintf(stderr, "ERROR - Couldn't open the kernel file\n");
		exit(EXIT_FAILURE);
	}
	kernel_bin[kernel_length] = 0;

	/* 1.7: Create program */
	program = clCreateProgramWithBinary(context, 1, &device_id, (const size_t *)&kernel_length, (const byte **)&kernel_bin, NULL, &err);
	OPENCL_CHECK_ERROR( err );

	err = clBuildProgram(program, 1, &device_id, "", NULL, NULL);
	if (err == CL_BUILD_PROGRAM_FAILURE) {
		size_t log_size;
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		char *log = (char *)malloc(log_size);
		MALLOC_CHECK( log );
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		printf("[OpenCL kernel build failure]\nLog dump:\n%s\n", log);
	}
	OPENCL_CHECK_ERROR( err );

	/* 1.8 Allocate and fetch kernels */
	kernels = (cl_kernel *)malloc(n_kernels * sizeof(cl_kernel));
	MALLOC_CHECK( kernels );
	for (int i = 0; i < n_kernels; ++i)
	{
		kernels[i] = clCreateKernel(program, entrypoints[i], &err);
		OPENCL_CHECK_ERROR( err );
	}

	OPENCL_CHECK( clReleaseProgram(program) );

	return kernels;
}

#ifdef __DO_NOT_COMPILE__

	/* 1.9: Create queues and allocate device buffers */
	properties = 0;
	queue = clCreateCommandQueue(context, device_id, properties, &err);
	OPENCL_CHECK_ERROR( err );

	d_curr = clCreateBuffer(context, CL_MEM_READ_ONLY, (1920 + 30) * (1088 + 30) * sizeof(byte), NULL, &err);
	OPENCL_CHECK_ERROR( err );
	d_prev = clCreateBuffer(context, CL_MEM_READ_ONLY, (1920 + 30) * (1088 + 30) * sizeof(byte), NULL, &err);
	OPENCL_CHECK_ERROR( err );

	d_SAD_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, results_per_frame * sizeof(unsigned short), NULL, &err);
	OPENCL_CHECK_ERROR( err );
	d_x_vector_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, results_per_frame * sizeof(int8), NULL, &err);
	OPENCL_CHECK_ERROR( err );
	d_y_vector_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, results_per_frame * sizeof(int8), NULL, &err);
	OPENCL_CHECK_ERROR( err );

	/* 1.10: Wait until all OpenCL operations are executed */
	OPENCL_CHECK( clFinish(queue) );

	/* 2: Compute motion estimation over the video's frames as they're read. */
	cl_event sync_events[3];

	/* 2.1: Copy frames to device */
	OPENCL_CHECK( clEnqueueWriteBuffer(queue, d_curr, CL_FALSE, 0,
		(1920 + 30) * (1088 + 30) * sizeof(byte), (void *)curr, 0, NULL, sync_events) );
	OPENCL_CHECK( clFlush(queue) );
	OPENCL_CHECK( clWaitForEvents(1, sync_events ) );

	OPENCL_CHECK( clEnqueueWriteBuffer(queue, d_prev, CL_FALSE, 0,
		(1920 + 30) * (1088 + 30) * sizeof(byte), (void *)prev, 0, NULL, sync_events) );
	OPENCL_CHECK( clFlush(queue) );
	OPENCL_CHECK( clWaitForEvents(1, sync_events ) );

	/* 2.2: Set kernel args */
	OPENCL_CHECK( clSetKernelArg(kernel_ME, 2, 16 * 16 * sizeof(cl_uchar), NULL) );
	OPENCL_CHECK( clSetKernelArg(kernel_ME, 3, 46 * 46 * sizeof(cl_uchar), NULL) );
	OPENCL_CHECK( clSetKernelArg(kernel_ME, 4, sizeof(cl_mem), &d_SAD_output) );
	OPENCL_CHECK( clSetKernelArg(kernel_ME, 5, sizeof(cl_mem), &d_x_vector_output) );
	OPENCL_CHECK( clSetKernelArg(kernel_ME, 6, sizeof(cl_mem), &d_y_vector_output) );

		/* ME over the whole frame */		
		OPENCL_CHECK( clEnqueueTask(queue, kernel_ME, 0, NULL, NULL) );
		OPENCL_CHECK( clFlush(queue) );

		/* 2.4: Read results from device */
		OPENCL_CHECK( clEnqueueReadBuffer(queue, d_SAD_output, CL_FALSE, 0,
			results_per_frame * sizeof(unsigned short), (void *)&SADs[frame_num * results_per_frame],
			0, NULL, sync_events) );
		OPENCL_CHECK( clEnqueueReadBuffer(queue, d_x_vector_output, CL_FALSE, 0,
			results_per_frame * sizeof(int8), (void *)&x_vectors[frame_num * results_per_frame],
			0, NULL, &sync_events[1]) );
		OPENCL_CHECK( clEnqueueReadBuffer(queue, d_y_vector_output, CL_FALSE, 0,
			results_per_frame * sizeof(int8), (void *)&y_vectors[frame_num * results_per_frame],
			0, NULL, &sync_events[2]) );
		OPENCL_CHECK( clFlush(queue) );
		OPENCL_CHECK( clWaitForEvents(3, sync_events) );

	/* 4: Free resources */
	OPENCL_CHECK( clReleaseMemObject(d_curr) );
	OPENCL_CHECK( clReleaseMemObject(d_prev) );
	OPENCL_CHECK( clReleaseMemObject(d_SAD_output) );
	OPENCL_CHECK( clReleaseMemObject(d_x_vector_output) );
	OPENCL_CHECK( clReleaseMemObject(d_y_vector_output) );

	OPENCL_CHECK( clReleaseKernel(kernel_ME) );
	OPENCL_CHECK( clReleaseCommandQueue(queue) );
	OPENCL_CHECK( clReleaseContext(context) );

	return 0;
}
#endif // __DO_NOT_COMPILE__

