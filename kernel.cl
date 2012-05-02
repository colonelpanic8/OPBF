#define LOCAL_WORK_SIZE 32
#define LWST2 64
#define MOD2 30
#define HALF_WARP 16
#define MIN(a,b) ((a) > (b) ? (b) : (a))

typedef struct _edge {
  uint source;
  uint dest;
  float weight;
} __attribute__ ((aligned (16))) edge;

typedef struct _vertex {
  uint num_edges;
  uint index;
} vertex;

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
  if(thread_id != 4)
    distances[thread_id] = INFINITY;
}


__kernel void UpdateVertex(
			   __global edge *edges,
			   __global float *distances,
			   __global uint *preds,
			   __global vertex *vertices,
			   __global uint *update,
			   uint num_vertices,
			   uint num_edges
)
{
  uint start = get_group_id(0) * LOCAL_WORK_SIZE;
  uint local_id = get_local_id(0);
  uint gid = get_global_id(0);
  uint __local current_edge[LOCAL_WORK_SIZE];
  int __local remaining_edges[LOCAL_WORK_SIZE];
  edge __local work[LOCAL_WORK_SIZE][HALF_WARP+1];
  vertex __local nodes[LOCAL_WORK_SIZE];
  int loading_id = local_id;
  uint offset = 0;
  uint i;
  float min;
  uint pred;
  bool __local did_update = 0;
  bool __local done = 0;
  if(gid == 0) update[0] = 0;
  if(local_id >= HALF_WARP) {
    loading_id = local_id - (HALF_WARP);
    offset = 1;
  }
  if(gid < num_vertices) {
    nodes[local_id] = vertices[gid];
    barrier(CLK_LOCAL_MEM_FENCE);
    remaining_edges[local_id] = nodes[local_id].num_edges;
    current_edge[local_id] = nodes[local_id].index;
    min = distances[start+local_id];
    pred = preds[start+local_id];
  } else {
    remaining_edges[local_id] = 0;
    min = INFINITY;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  while(!done) {
    for(i = 0; i < LOCAL_WORK_SIZE; i += 2) {
      if(remaining_edges[i+offset] > loading_id) {
	work[i+offset][loading_id] = edges[current_edge[i+offset]+loading_id];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint max = MIN(HALF_WARP,remaining_edges[local_id]);
    for(i = 0; i < max; i++) {
      float temp = distances[work[local_id][i].source];
      if(temp < INFINITY) {
	temp = work[local_id][i].weight + temp;
	if(min > temp) {
	    did_update = 1;
	    min = temp;
	    pred = work[local_id][i].source;
	}
      }
    }
    current_edge[local_id] += HALF_WARP;
    remaining_edges[local_id] -= HALF_WARP;
    if(local_id == 0)
      done = 1;
    if(remaining_edges[local_id] > 0)
      done = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(did_update) {
    distances[start+local_id] = min;
    preds[start+local_id] = pred;
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
