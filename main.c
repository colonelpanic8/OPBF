#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <getopt.h>
#include <time.h>
#include <sys/time.h>

/*--------------------------------------------------------------------------------*/

#ifdef __APPLE__
    #include <OpenCL/opencl.h>
    #include <mach/mach_time.h>
    typedef uint64_t                    time_delta_t;
    typedef mach_timebase_info_data_t   frequency_t;
#else
    #include <CL/cl.h>
    typedef struct timeval              time_delta_t;
    typedef double                      frequency_t;
#endif

/*--------------------------------------------------------------------------------*/

typedef int (*Compare_fn)(const void *, const void *);
typedef struct _edge {
  cl_uint source;
  cl_uint dest;
  cl_float weight;
} __attribute__ ((aligned (16))) edge;

typedef struct _gpu_edge {
  cl_uint source;
  cl_float weight;
} cl_edge;

typedef struct _vertex {
  cl_uint num_edges;
  cl_uint index;
  cl_uint spot;  
} __attribute__ ((aligned (16))) vertex;

typedef struct _gpu_vertex {
  cl_uint num_edges;
  cl_uint index;
} cl_vertex;



#define PRINT
//#define DOMM
#define LOCAL_WORK_SIZE 32
#define MAX_RANDOM_FLOAT 10
#define DEFAULT_NUM_VERTICES 32
#define DEFAULT_NUM_EDGES 16*DEFAULT_NUM_VERTICES

#define DEFAULT_KERNEL_FILENAME ("kernel.cl")
#define problem(...) fprintf(stderr, __VA_ARGS__)
#define BAR "--------------------------------------------------------------------------------\n"
#define PRINT_ROW_LENGTH 16

/*--------------------------------------------------------------------------------*/

static const char*
GetErrorString(cl_int error) {
    switch(error)
    {
    case(CL_SUCCESS):                           return "Success";
    case(CL_DEVICE_NOT_FOUND):                  return "Device not found!";
    case(CL_DEVICE_NOT_AVAILABLE):              return "Device not available!";
    case(CL_MEM_OBJECT_ALLOCATION_FAILURE):     return "Memory object allocation failure!";
    case(CL_OUT_OF_RESOURCES):                  return "Out of resources!";
    case(CL_OUT_OF_HOST_MEMORY):                return "Out of host memory!";
    case(CL_PROFILING_INFO_NOT_AVAILABLE):      return "Profiling information not available!";
    case(CL_MEM_COPY_OVERLAP):                  return "Overlap detected in memory copy operation!";
    case(CL_IMAGE_FORMAT_MISMATCH):             return "Image format mismatch detected!";
    case(CL_IMAGE_FORMAT_NOT_SUPPORTED):        return "Image format not supported!";
    case(CL_INVALID_VALUE):                     return "Invalid value!";
    case(CL_INVALID_DEVICE_TYPE):               return "Invalid device type!";
    case(CL_INVALID_DEVICE):                    return "Invalid device!";
    case(CL_INVALID_CONTEXT):                   return "Invalid context!";
    case(CL_INVALID_QUEUE_PROPERTIES):          return "Invalid queue properties!";
    case(CL_INVALID_COMMAND_QUEUE):             return "Invalid command queue!";
    case(CL_INVALID_HOST_PTR):                  return "Invalid host pointer address!";
    case(CL_INVALID_MEM_OBJECT):                return "Invalid memory object!";
    case(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR):   return "Invalid image format descriptor!";
    case(CL_INVALID_IMAGE_SIZE):                return "Invalid image size!";
    case(CL_INVALID_SAMPLER):                   return "Invalid sampler!";
    case(CL_INVALID_BINARY):                    return "Invalid binary!";
    case(CL_INVALID_BUILD_OPTIONS):             return "Invalid build options!";
    case(CL_INVALID_PROGRAM):                   return "Invalid program object!";
    case(CL_INVALID_PROGRAM_EXECUTABLE):        return "Invalid program executable!";
    case(CL_INVALID_KERNEL_NAME):               return "Invalid kernel name!";
    case(CL_INVALID_KERNEL):                    return "Invalid kernel object!";
    case(CL_INVALID_ARG_INDEX):                 return "Invalid index for kernel argument!";
    case(CL_INVALID_ARG_VALUE):                 return "Invalid value for kernel argument!";
    case(CL_INVALID_ARG_SIZE):                  return "Invalid size for kernel argument!";
    case(CL_INVALID_KERNEL_ARGS):               return "Invalid kernel arguments!";
    case(CL_INVALID_WORK_DIMENSION):            return "Invalid work dimension!";
    case(CL_INVALID_WORK_GROUP_SIZE):           return "Invalid work group size!";
    case(CL_INVALID_GLOBAL_OFFSET):             return "Invalid global offset!";
    case(CL_INVALID_EVENT_WAIT_LIST):           return "Invalid event wait list!";
    case(CL_INVALID_EVENT):                     return "Invalid event!";
    case(CL_INVALID_OPERATION):                 return "Invalid operation!";
    case(CL_INVALID_GL_OBJECT):                 return "Invalid OpenGL object!";
    case(CL_INVALID_BUFFER_SIZE):               return "Invalid buffer size!";
    case(CL_COMPILER_NOT_AVAILABLE):            return "Compiler not available!";
    case(CL_BUILD_PROGRAM_FAILURE):             return "Build failure!";
    default:                                    return "Unknown error!";
    };
    return "Unknown error";
}

/*--------------------------------------------------------------------------------*/

static unsigned long ReadFromTextFile(FILE *fh, char* buffer, size_t buffer_size) {
    unsigned long count = (unsigned long)fread(buffer, buffer_size, 1, fh);
    buffer[buffer_size] = '\0';
    return count;
}

static char *LoadTextFromFile(const char *filename, unsigned long *size /* returned file size in bytes */) {
  FILE* fh;
  struct stat statbuf;
  
  //Open File.
  fh = fopen(filename, "r");
  if (!fh)
    problem("File did not open successfully\n");
  
  //Get file size.
  stat(filename, &statbuf);
  unsigned long bytes = (*size);
  
  if(size)
    (*size) = (unsigned long)statbuf.st_size;
  bytes = *size;

  //To be returned.
  char *text = (char*)malloc(*size + 1);
  if(!text)
    return 0;
  
  ReadFromTextFile(fh, text, bytes);
  fclose(fh);
  
  return text;
}

/*--------------------------------------------------------------------------------*/

void check_failure(cl_int err) {
  if (err != CL_SUCCESS) {
    problem("%s", GetErrorString(err));
    exit(err);
  }
}

void printArray(cl_float *matrix, cl_int num) {
  int i,j;
  for(i = 0; i < num; i += PRINT_ROW_LENGTH) {
    for(j = 0; j < PRINT_ROW_LENGTH; j++) {
      if(i+j > num)
	break;
      printf("%8d ", i+j);
    }
    printf("\n");
    for(j = 0; j < PRINT_ROW_LENGTH; j++) {
      if(i+j > num)
	break;
      printf("%8.0f ", matrix[i+j]);
    }
    printf("\n");
    printf(BAR);
  }
}

void UIprintArray(cl_uint *matrix, cl_int num) {
  int i,j;
  for(i = 0; i < num; i += PRINT_ROW_LENGTH) {
    for(j = 0; j < PRINT_ROW_LENGTH; j++) {
      if(i+j > num)
	break;
      printf("%4d ", i+j);
    }
    printf("\n");
    for(j = 0; j < PRINT_ROW_LENGTH; j++) {
      if(i+j > num)
	break;
      printf("%4d ", matrix[i+j]);
    }
    printf("\n");
    printf(BAR);
  }
}

void generate_graph(cl_uint *vertex_index, cl_uint *edge_count,
		    cl_uint *edge_sources, float *edge_weights, cl_uint num_vertices, 
		    cl_uint num_edges) 
{
  cl_uint edges_per_vertex = num_edges/num_vertices;
  int n = 0;
  cl_uint i, j;
  srand(time(NULL));
  for(i = 0; i < num_vertices; i++) {
    vertex_index[i] = n;
    for(j = 0; j < edges_per_vertex; j++) {
      edge_sources[n+j] = rand() % num_vertices;
      edge_weights[n+j] = rand() % 1000;
    }
    n += j;
    edge_count[i] = j;
  }
  
}

void generate_graph2(cl_uint *vertex_index, cl_uint *edge_count,
		    cl_uint *edge_sources, float *edge_weights, cl_uint num_vertices, 
		    cl_uint num_edges) 
{
  cl_uint edges_per_vertex = num_edges/num_vertices;
  int n = 0;
  cl_uint i, j;
  srand(time(NULL));
  for(i = 0; i < num_vertices; i++) {
    vertex_index[i] = n;
    for(j = 0; j < edges_per_vertex; j++) {
      edge_sources[n+j] = rand() % num_vertices;
      edge_weights[n+j] = rand() % 1000;
    }
    n += j;
    edge_count[i] = j;
  }
}

void print_edges(edge *edges, cl_uint num) {
  cl_uint i,j;
  for(i = 0; i < num; i += PRINT_ROW_LENGTH) {
    for(j = 0; j < PRINT_ROW_LENGTH; j++) {
      if(i+j >= num)
	break;
      printf("%7d     ", i+j);
    }
    printf("\n");
    for(j = 0; j < PRINT_ROW_LENGTH; j++) {
      if(i+j >= num)
	break;
      printf("%3u %3u %3.0f|", edges[i+j].dest, edges[i+j].source, edges[i+j].weight);
    }
    printf("\n");
    printf(BAR);
  }
}

int destcomp(const void *a, const void *b) {
  edge *f = (edge *) a;
  edge *s = (edge *) b;
  return f->dest - s->dest;
}

cl_uint graph_data_from_file(char *filename, edge **res){
  FILE *file;
  char buffer[256];
  size_t i;
  edge *edges;
  file = fopen(filename, "r");
  buffer[0] = '\0';

  while(buffer[0] != 'a')
    fgets(buffer, 256, file);
  for(i = 1; fgets(buffer, 256, file); i++);
  edges = (edge *)malloc(sizeof(edge)*i);
  fseek(file, 0, SEEK_SET);
  while(buffer[0] != 'a')
    fgets(buffer, 256, file);
  i = 0;
  buffer[0] = '\0';
  while(buffer[0] != 'a')
    fgets(buffer, 256, file);
  do {
    sscanf(buffer, "a %u %u %f", &(edges[i].dest), &(edges[i].source), &(edges[i].weight));
    edges[i].dest -= 1;
    edges[i].source -= 1;
    i++;
  } while(fgets(buffer, 256, file));
  mergesort(edges, i, sizeof(edge), destcomp);
  *res = edges;
  return i;
}

int countcomp(const void *a, const void *b) {
  vertex *f = (vertex *) a;
  vertex *s = (vertex *) b;
  return s->num_edges - f->num_edges;
}

cl_uint build_vertex_array(edge *edges, cl_uint edge_count, vertex **v) {
  cl_uint max = edges[edge_count - 1].dest;
  vertex *vertices = (vertex *)malloc(sizeof(vertex)*max);
  cl_uint i,j, last = 0;
  j = 0;
  i = edges[j].dest;
  while(j < edge_count) {
    while(edges[j].dest == i && j < edge_count) {
      j++;
    }
    vertices[i].num_edges = j - last;
    vertices[i].index = last;
    vertices[i].spot = i;
    last = j;
    i = edges[j].dest;
  }
  mergesort(vertices, max, sizeof(vertex), countcomp);
  *v = vertices;
  return max;
}

cl_uint *make_map(vertex *v, cl_uint max) {
  cl_uint i;
  cl_uint *o = (cl_uint *)malloc(sizeof(cl_uint)*max);
  for(i = 0; i < max; i++) {
    o[v[i].spot] = i;
  }
  return o;
}

cl_edge *make_gpu_edges(edge *edges, cl_uint num_edges) {
  cl_uint i;
  cl_edge *res = (cl_edge *)malloc(sizeof(cl_edge)*num_edges);
  for(i = 0; i < num_edges; i++) {
    res[i].source = edges[i].source;
    res[i].weight = edges[i].weight;
  }
  return res;
}

cl_vertex *make_gpu_vertices(vertex *vertices, cl_uint num_vertices) {
  cl_uint i;
  cl_vertex *res = (cl_vertex *)malloc(sizeof(cl_vertex)*num_vertices);
  for(i = 0; i < num_vertices; i++) {
    res[i].num_edges = vertices[i].num_edges;
    res[i].index = vertices[i].index;
  }
  return res;
}

/*--------------------------------------------------------------------------------*/

struct timeval tv_delta(struct timeval start, struct timeval end){
  struct timeval delta = end;
  delta.tv_sec -= start.tv_sec;
  delta.tv_usec -= start.tv_usec;
  if (delta.tv_usec < 0) {
    delta.tv_usec += 1000000;
    delta.tv_sec--;
  }
  return delta;
}

/*--------------------------------------------------------------------------------*/

cl_int *getMatrixFromFile(char *filename, cl_int *size) {
  FILE *fh;
  cl_int *output;
  int i, matrix_size;

  fh = fopen(filename, "r");
  if(!fh)
    problem("file failed to open\n");
  
  fscanf(fh, "%d", &matrix_size);
  *size = matrix_size;
  output = (cl_int *)malloc(sizeof(cl_int)*matrix_size*matrix_size);

  if(!output)
    exit(-1);
  for(i = 0; i < matrix_size*matrix_size; i++) {
      fscanf(fh, "%d", &(output[i]));
  }
  return output;
}

cl_float *randomMatrix(int size) {
  int num = size*size;
  cl_float *output = (cl_float *)malloc(sizeof(cl_float)*num);
  int i, j;
  for(i = 0; i < size; i++) {
    output[i] = INFINITY;
  }
  for(i = 0; i < size; i++) {
    memmove(output + size*i, output, size*sizeof(cl_float));
  }
  srand(time(NULL));
  for(i = 0; i < size; i++) {
    for(j = 0; j < size/4; j++) {
      int index = rand() % size;
      output[size*i + index] = rand() % 400;
    }
    output[size*i + i] = 0;
  }
  return output;
}

cl_uint *initPreds(int size) {
  cl_uint *output = (cl_uint *)malloc(sizeof(cl_uint)*size*size);
  int i;
  for(i = 0; i < size*size; i++) {
    output[i] = i/size + 1;
  }
  return output;
}

void printMatrix(cl_float *matrix, cl_int rows, cl_int cols) {
  int i;
  printf("%4.0f ", (float)-1); 
  for(i = 0; i < cols; i++)
    printf("%4.0d ",i+1); 
  printf("\n");
  for(i=0; i<cols; i++)
    printf("_____");
  printf("____");
    
  
  for(i = 0; i < rows*cols; i++) {
    if(i % cols == 0) {
      printf("\n");
      printf("%4d|", (i/cols + 1));
    }
    if(matrix[i] > 4000000)
      matrix[i] = INFINITY;
    printf("%4.0f ", matrix[i]);
  }
  printf("\n");
}

void printPreds(cl_uint *matrix, cl_int size) {
  int i;
  printf("%4.0f ", (float)-1); 
  for(i = 0; i < size; i++)
    printf("%4.0d ", i+1); 
  printf("\n");
  for(i=0; i<size; i++)
    printf("_____");
  printf("____");
  for(i = 0; i < size*size; i++) {
    if(i % size == 0) {
      printf("\n");
      printf("%4d|", (i/size + 1));
    }
    printf("%4d ", matrix[i]);
  }
  printf("\n");
}

cl_float *generate_matrix(cl_uint *vertex_index, cl_uint *edge_count,
			  cl_uint *edge_sources, float *edge_weights, cl_uint num_vertices, 
			  cl_uint num_edges) {
  cl_uint i,j,k;
  cl_float *output = (cl_float*)malloc(num_vertices*num_vertices*sizeof(cl_float));
  for(i = 0; i < num_vertices; i++) {
    output[i] = INFINITY;
  }
  for(i = 0; i < num_vertices; i++) {
    memmove(output + num_vertices*i, output, num_vertices*sizeof(cl_float));
  }
  for(i = 0; i < num_vertices; i++) {
    k = vertex_index[i]; 
    for(j = 0; j < edge_count[i]; j++) {
      int source = edge_sources[k+j];
      float weight = edge_weights[k+j];
	if(output[source*num_vertices + i] > weight)
	  output[source*num_vertices + i] = weight;
    }
    output[num_vertices*i + i] = 0;
  }
  return output;
}

/*--------------------------------------------------------------------------------*/
#define BLOCK_SIZE 16

cl_float *matrix_multiply(cl_program program, cl_context context, cl_command_queue commands, cl_float *matrix,
		     cl_uint nv) {
  cl_int err;
  cl_kernel kernel;
  cl_int m_size = nv;
  kernel = clCreateKernel(program, "matrix_product", &err);
  check_failure(err);
  
  printf(BAR);
  //Read matrices from file
  cl_uint *preds_init;
  cl_mem input;
  cl_mem input2;
  cl_mem preds;
  preds_init = initPreds(m_size);
   
  
#ifdef PRINT
  if(m_size < 100) {
    printMatrix(matrix, m_size, m_size);
    printf(BAR);
    printPreds(preds_init, m_size);
  }
#endif
  
  printf("Creating data buffers.\n");
  printf(BAR);
  //Create data buffers on the device.
  input = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(cl_float)*m_size*m_size, NULL, NULL);

  input2 = clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(cl_float)*m_size*m_size, NULL, NULL);
  
  preds = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(cl_uint)*m_size*m_size, NULL, NULL);
  
  if(!input || !input2 || !preds ) {
    problem("Failed to allocate device memory.\n");
    exit(-1);
  }
  
  err = 0;
  printf("Putting data into device memory.\n");
  printf(BAR);
  //Put data into device Memory.
  err  =  clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, 
  		       sizeof(cl_float)*m_size*m_size, matrix, 0, NULL, NULL);
  err  =  clEnqueueWriteBuffer(commands, preds, CL_TRUE, 0, 
  		       sizeof(cl_uint)*m_size*m_size, preds_init, 0, NULL, NULL);
  check_failure(err);
  
  printf("Setting Kernel Arguments.\n");
  printf(BAR);
  //Set arguments.
  int i = 0;
  err  =  clSetKernelArg(kernel, i++, sizeof(cl_mem), &input);
  err |=  clSetKernelArg(kernel, i++, sizeof(cl_mem), &input2);
  err |=  clSetKernelArg(kernel, i++, sizeof(cl_mem), &preds);
  err |=  clSetKernelArg(kernel, i++, sizeof(cl_int), &m_size);
  check_failure(err);

  //Determine work group size.
  size_t global_size[] = {m_size, m_size};
  size_t local_size[] = {BLOCK_SIZE, BLOCK_SIZE};
  
  printf("Running.\n");
  printf(BAR);
  clFinish(commands);
  //Run our program.
  struct timeval start, end, delta;
  gettimeofday(&start, NULL);
  int exp = 1;
  while(exp < m_size) {
    err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
    check_failure(err);
    clFinish(commands);
    exp = exp*2;
    i = 0;
    clSetKernelArg(kernel, i++, sizeof(cl_mem), &input2);
    clSetKernelArg(kernel, i, sizeof(cl_mem), &input);
    clFinish(commands);
    err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
    clFinish(commands);
    exp = exp*2;
    i = 0;
    clSetKernelArg(kernel, i++, sizeof(cl_mem), &input);
    clSetKernelArg(kernel, i, sizeof(cl_mem), &input2);
    printf("%d\n", exp);
    clFinish(commands);
  }
  gettimeofday(&end, NULL);
  delta = tv_delta(start, end);
  printf("GPU Time: %ld.%06ld\n", 
	 (long int)delta.tv_sec, 
	 (long int)delta.tv_usec);
  printf(BAR);
  
  printf("Getting data.\n");
  printf(BAR);
  //Retrieve and print output.
  cl_float *result;
  cl_uint *preds_result;
  result = (cl_float *)malloc(sizeof(cl_float)*m_size*m_size);
  preds_result = (cl_uint *)malloc(sizeof(cl_int)*m_size*m_size);
  err = clEnqueueReadBuffer(commands, input2, CL_TRUE, 0, sizeof(cl_float)*m_size*m_size,
			    result, 0, NULL, NULL );
  err = clEnqueueReadBuffer(commands, preds, CL_TRUE, 0, sizeof(cl_uint)*m_size*m_size,
			    preds_result, 0, NULL, NULL );
  check_failure(err);
  
  /*
  printMatrix(matrix, m_size, m_size);
  printf("*\n");
  printMatrix(right_matrix, m_size, m_size);
  printf("=\n");
  printMatrix(result, m_size, m_size);
  */

  //Do the computation on the CPU to verify.
  
#ifdef PRINT
  if(m_size < 100) {
    printMatrix(result, m_size, m_size);
    printf(BAR);
    printPreds(preds_result, m_size);
  }
#endif


  printf(BAR);
  printf("Cleanup.\n");
  //Device Cleanup.
  clReleaseKernel(kernel);
  clReleaseMemObject(input);
  clReleaseMemObject(input2);

  //Memory Cleanup.
  free(result);
  free(preds_result);
  return matrix;
}

/*--------------------------------------------------------------------------------*/

int main(int argc, char **argv) {
  cl_int err;

  //Get device id.
  cl_device_id device_id;
  cl_platform_id platform = 0;
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU , 1, &device_id, NULL);
  check_failure(err);
  
  //Output the name of our device.
  cl_char vendor_name[1024];
  cl_char device_name[1024];
  err = clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(vendor_name), vendor_name, NULL);
  err|= clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
  check_failure(err);
  printf("Using %s %s. \n", vendor_name, device_name);
  printf(BAR);
  
  //Create a context.
  cl_context context;
  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  check_failure(err);

  //Create a command queue.
  cl_command_queue commands;
  commands = clCreateCommandQueue(context, device_id, 0, &err);
  check_failure(err);

  //Load kernel from file into a string.
  char *source;
  unsigned long source_length = 0;
  if(argc < 2) {
    source = LoadTextFromFile(DEFAULT_KERNEL_FILENAME, &source_length);
  } else {
    source = LoadTextFromFile(argv[1], &source_length);
  }
  
  //Create our kernel.
  cl_program program;
  cl_kernel update_vertex_kernel;
  cl_kernel init_distances_kernel;
  program = clCreateProgramWithSource(context, 1, (const char **)&source, NULL, &err);
  check_failure(err);
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    char buffer[9999];
    
    problem("ERROR: Failed to build program executable! %s\n", GetErrorString(err));
    err = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
				sizeof(buffer), buffer, NULL);
    check_failure(err);
    problem("okay...%s\n", buffer);
    return EXIT_FAILURE;
  }
  update_vertex_kernel = clCreateKernel(program, "UpdateVertex", &err);
  init_distances_kernel = clCreateKernel(program, "InitDistances", &err);
  check_failure(err);
  
  //

  edge *edges;
  vertex *vertices;
  char buffer[256] = "NewYorkRM";
  char *name;
  if(argc > 1) {
    name = argv[1];
  } else{
    name = buffer;
  }
  
  
  cl_uint num_edges = graph_data_from_file(name, &edges);
  cl_uint num_vertices  = build_vertex_array(edges, num_edges, &vertices);
  cl_uint *map = make_map(vertices, num_vertices);
  cl_edge *device_edges = make_gpu_edges(edges, num_edges);
  cl_vertex *device_vertices = make_gpu_vertices(vertices, num_vertices);
  
  cl_float *distances = (cl_float *)malloc(sizeof(cl_float)*num_vertices);
  
  cl_mem _edges;
  cl_mem _vertices;
  cl_mem _distances;
  cl_mem _update;
  cl_mem _preds;
  cl_mem _map;
  

  printf("Creating data buffers.\n");
  printf(BAR);
  //Create data buffers on the device.
  
  
  _distances    = clCreateBuffer(context, CL_MEM_READ_WRITE,
				 sizeof(cl_float)*num_vertices, NULL, NULL);
  _preds        = clCreateBuffer(context, CL_MEM_READ_WRITE,
				 sizeof(cl_uint)*num_vertices, NULL, NULL);
  _edges        = clCreateBuffer(context,  CL_MEM_READ_ONLY,
				 sizeof(edge)*num_edges,   NULL, NULL);
  _vertices     = clCreateBuffer(context,  CL_MEM_READ_ONLY,
				 sizeof(vertex)*num_vertices, NULL, NULL);
  _map          = clCreateBuffer(context,  CL_MEM_READ_ONLY,
				 sizeof(cl_uint)*num_vertices, NULL, NULL);
  _update       = clCreateBuffer(context, CL_MEM_READ_WRITE,
				 sizeof(cl_uint), NULL, NULL);

  
  if(!_vertices || !_edges || !_distances || !_update || !_map) {
    problem("Failed to allocate device memory.\n");
    exit(-1);
  }
  
  printf("Putting data into device memory.\n");
  printf(BAR);
  //Put data into device Memory.
  err  =  clEnqueueWriteBuffer(commands, _edges, CL_TRUE, 0, 
			       sizeof(cl_edge)*num_edges , device_edges, 0, NULL, NULL);
  err |=  clEnqueueWriteBuffer(commands, _vertices, CL_TRUE, 0,
			       sizeof(cl_vertex)*num_vertices, device_vertices, 0, NULL, NULL);
  err |=  clEnqueueWriteBuffer(commands, _map, CL_TRUE, 0,
			       sizeof(cl_uint)*num_vertices, map, 0, NULL, NULL);

  check_failure(err);


  
  int a = 0;
  printf("Setting Kernel Arguments.\n");
  printf(BAR);
  //Set arguments.
  err  =  clSetKernelArg(init_distances_kernel, 0, sizeof(cl_mem), &_distances);


  err  =  clSetKernelArg(update_vertex_kernel, a++, sizeof(cl_mem), &_edges);
  err |=  clSetKernelArg(update_vertex_kernel, a++, sizeof(cl_mem), &_distances);
  err |=  clSetKernelArg(update_vertex_kernel, a++, sizeof(cl_mem), &_preds);
  err |=  clSetKernelArg(update_vertex_kernel, a++, sizeof(cl_mem), &_vertices);
  err |=  clSetKernelArg(update_vertex_kernel, a++, sizeof(cl_mem), &_update);
  err |=  clSetKernelArg(update_vertex_kernel, a++, sizeof(cl_mem), &_map);
  err |=  clSetKernelArg(update_vertex_kernel, a++, sizeof(cl_uint), &num_vertices);
  err |=  clSetKernelArg(update_vertex_kernel, a++, sizeof(cl_uint), &num_edges);
  
  check_failure(err);

  clFinish(commands);
  printf("Running.\n");
  printf(BAR);
  
  size_t global[] = {num_vertices + LOCAL_WORK_SIZE - (num_vertices % LOCAL_WORK_SIZE)};
  size_t local[] = {LOCAL_WORK_SIZE};
  //Run our program.


  clEnqueueNDRangeKernel(commands, init_distances_kernel, 1, NULL, global, NULL, 0, NULL, NULL);
  cl_float *result;
  result = (cl_float *)malloc(sizeof(cl_float)*num_vertices); 
  clFinish(commands);
  cl_uint i;
  cl_uint update;
  //cl_uint *preds = (cl_uint *)malloc(sizeof(cl_uint)*num_vertices);
  struct timeval start, end, delta;
  gettimeofday(&start, NULL);
  for(i = 0; i < num_vertices; i++) {
    /*
    err = clEnqueueReadBuffer(commands, _distances, CL_TRUE, 0, sizeof(cl_float)*128,
			      result, 0, NULL, NULL );
    err = clEnqueueReadBuffer(commands, _preds, CL_TRUE, 0, sizeof(cl_float)*128,
			      preds, 0, NULL, NULL );
    clFinish(commands);
    //    printArray(result, 64);
    //UIprintArray(preds, 64);
    */
    err = clEnqueueNDRangeKernel(commands, update_vertex_kernel, 1, NULL, global, local, 0, NULL, NULL);
    clFinish(commands);
    err = clEnqueueReadBuffer(commands, _update, CL_TRUE, 0, sizeof(cl_uint),
			      &update, 0, NULL, NULL );
    clFinish(commands);
    printf("Round %d, update: %d \n", i, update);
    if(!update) break;
  }
  err = clEnqueueReadBuffer(commands, _distances, CL_TRUE, 0, sizeof(cl_float)*num_vertices,
				result, 0, NULL, NULL );
  clFinish(commands);
  printArray(result, 64);
  check_failure(err);
  clFinish(commands);
  gettimeofday(&end, NULL);
  delta = tv_delta(start, end);
  printf("GPU Time: %ld.%06ld\n", 
	 (long int)delta.tv_sec, 
	 (long int)delta.tv_usec);
  printf(BAR);
  
  printf("Getting data.\n");
  printf(BAR);
  //Retrieve and print output.
  err = clEnqueueReadBuffer(commands, _distances, CL_TRUE, 0, sizeof(cl_float)*num_vertices,
			    result, 0, NULL, NULL );
  //printArray(result, num_vertices);
  check_failure(err);
  
  

  //Do the computation on the CPU to verify.
  printf("Cleanup.\n");
  //Device Cleanup.
  clReleaseProgram(program);
  clReleaseKernel(update_vertex_kernel);
  clReleaseKernel(init_distances_kernel);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);
  clReleaseMemObject(_distances);
  clReleaseMemObject(_vertices);
  clReleaseMemObject(_edges);

  //Memory Cleanup.
  free(edges);
  free(vertices);
  free(distances);  
  
  return 0;
}
