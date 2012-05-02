#define LOCAL_WORK_SIZE 32
#define LWST2 64
#define MOD2 30
#define HALF_WARP 16
#define MIN(a,b) ((a) > (b) ? (b) : (a))

typedef struct _edge {
  cl_uint source;
  cl_uint dest;
  cl_float weight;
} edge;

void LoadGlobalToLocalf(float __global *g, float __local *l, uint width, uint id);
void LoadGlobalToLocali(uint __global *g, uint __local *l, uint width, uint id);

inline void LoadGlobalToLocalf(float __global *g, float __local *l, uint width, uint id) {
  l[id] = (id < width) ? g[id] : INFINITY;
}

inline void LoadGlobalToLocali(uint __global *g, uint __local *l, uint width, uint id) {
  l[id] = (id < width) ? g[id] : 0;
}

__kernel void InitDistances(__global float *distances)
{
  uint thread_id = get_global_id(0);
  if(thread_id)
    distances[thread_id] = INFINITY;
}


__kernel void UpdateVertex(
			   __global uint *edge_source,
			   __global float *edge_weights,
			   __global float *distances,
			   __global uint *vertex_index, //where the edges for a given vertex start in the edges arrays
			   __global uint *num_edges, //is this neccesary? should it be computed with vertex_index
			   __global uint *update
)
{
  
  uint start = get_group_id(0) * LOCAL_WORK_SIZE;
  uint local_id = get_local_id(0);
  uint gid = get_global_id(0);
  uint __local current_edge[LOCAL_WORK_SIZE];
  int __local remaining_edges[LOCAL_WORK_SIZE];
  float __local e_weights[LOCAL_WORK_SIZE][HALF_WARP+1]; //since threads access this array columnwise
  uint __local  e_sources[LOCAL_WORK_SIZE][HALF_WARP+1];
  uint loading_id = local_id;
  uint offset = 0;
  uint i;
  float min = distances[start+local_id];
  bool __local did_update = 0;
  if(gid == 0) update[0] = 0;
  if(local_id >= HALF_WARP) {
    loading_id = local_id - (HALF_WARP);
    offset = 1;
  }
  current_edge[local_id] = vertex_index[gid];
  remaining_edges[local_id] = num_edges[gid];
  barrier(CLK_LOCAL_MEM_FENCE);

  while(remaining_edges[0] > 0) {
    for(i = 0; i < LOCAL_WORK_SIZE; i += 2) {
      if(remaining_edges[i+offset] > loading_id) {
	e_weights[i+offset][loading_id] = edge_weights[current_edge[i+offset]+loading_id];
	e_sources[i+offset][loading_id] = edge_source[current_edge[i+offset]+loading_id];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint max = MIN(HALF_WARP,remaining_edges[local_id]);
    for(i = 0; i < max; i++) {
      float temp = distances[e_sources[local_id][i]];
      if(temp < INFINITY) {
	temp = e_weights[local_id][i] + temp;
	if(min > temp) {
	  did_update = 1;
	  min = temp;
	}
      }
    }
    current_edge[local_id] += HALF_WARP;
    remaining_edges[local_id] -= HALF_WARP;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(did_update) {
    distances[start+local_id] = min;
    if(local_id == 0) update[0] = 1;
  }
}

#define BLOCK_SIZE 16
#define index(y, x, the_size) (y*the_size + x)

void __kernel matrix_product(
			    float __global *matrix,
			    float __global *results,
			    uint __global *preds,
			    int size
			     ) { 
  //left matrix is shared_size x height, right matrix is width x shared_size
  float __local  l_block[BLOCK_SIZE][BLOCK_SIZE], r_block[BLOCK_SIZE][BLOCK_SIZE+1];
  int __local pr_block[BLOCK_SIZE][BLOCK_SIZE];
  //The l_block matrix is padded because we will access its elements column wise.
  //Padding the r_block matrix is unnecessary, since its elements are only accessed by row
  int i, j, blockR, blockC, r, c;
  blockC = get_group_id(0) * BLOCK_SIZE;
  blockR = get_group_id(1) * BLOCK_SIZE;
  c = get_local_id(0);
  r = get_local_id(1);
  float weight;
  int pred = preds[index(get_global_id(1), get_global_id(0), size)];
  weight = INFINITY;


  for(i = 0; i*BLOCK_SIZE < size; i++) { //try using a variable to store shared_size/16?
    //load subblock into local memory
    /*
    left += BLOCK_SIZE;
    right += size*BLOCK_SIZE;
    l_block[r][c] = 
      matrix[left + r*size + c];

    r_block[r][c] = 
    matrix[right + r*size + c];*/
    l_block[r][c] = 
      *(matrix + (blockR+r)*size + (i * BLOCK_SIZE) + c);
    r_block[r][c] = 
      *(matrix + blockC + (i * size * BLOCK_SIZE) + r*size + c);
    pr_block[r][c] = 
      *(preds + blockC + (i * size * BLOCK_SIZE) + r*size + c);
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for(j = 0; j< BLOCK_SIZE; j++) {
      int temp = l_block[r][j]+r_block[j][c];
      if(temp < weight) {
	weight = temp;
	pred = pr_block[j][c];
      }
    }
    
    //Make sure that we are done with the matrices stored in local memory before we enter the next iteration
    barrier(CLK_LOCAL_MEM_FENCE); 
  }
  
  results[get_global_id(1)*size + get_global_id(0)] = weight;
  preds[get_global_id(1)*size + get_global_id(0)] = pred;
  //preds[index(get_global_id(1), get_global_id(0), size)] = pred;
}
