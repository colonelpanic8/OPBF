#define LOCAL_WORK_SIZE 32
#define LWST2 64
#define MOD2 30
#define HALF_WARP 16
#define MIN(a,b) ((a) > (b) ? (b) : (a))

/*__kernel void UpdateVertex(
__global uint *edge_source,
__global float *edge_weights,
__global float *distances,
__global uint *vertex_index, //where the edges for a given vertex start in the edges arrays
__global uint *num_edges //is this neccesary; should it be computed with vertex_index
			   )
{
  uint start = get_group_id(0) * LOCAL_WORK_SIZE;
  uint local_id = get_local_id(0);
  int to_load, i;
  int rem_edges;
  float __local e_weights[LOCAL_WORK_SIZE];
  float __local s_weights[LOCAL_WORK_SIZE];
  float __local updates[LOCAL_WORK_SIZE];
  float __local mins[LOCAL_WORK_SIZE];

  for(i = 0; i < LOCAL_WORK_SIZE; i++) {
    uint item = start + i;
    float min = INFINITY;

    to_load = vertex_index[item];
    rem_edges = num_edges[item];
   
    while(rem_edges > 0) {
      uint source = edge_source[to_load + local_id];
      LoadGlobalToLocal(edge_weights + to_load, e_weights, rem_edges, local_id);
      s_weights[local_id] = distances[source];
      updates[local_id] = e_weights[local_id] + s_weights[local_id];
      
      barrier(CLK_LOCAL_MEM_FENCE); //This might not actually be necessary with LOCAL_WORK_SIZE = 32
      
      if(!(local_id & 1))
	ComputeMin(updates, MIN(rem_edges, LOCAL_WORK_SIZE), local_id);
      
      if(min > updates[0])
	min = updates[0];
      rem_edges -= LOCAL_WORK_SIZE;
      to_load += LOCAL_WORK_SIZE;  //talk to jim about this?
    }
    mins[i] = min; //Only one thread needs to do this.  Maybe have thread 0 should be in charge?
    //In any case, I believe the desired behavior results
  }
  distances[start+local_id] = mins[local_id];
}


void ComputeMin(float *values, int width, uint id) {
  uint stride = 1;
  uint shift = 0;
  while(stride < width) {
    if((id >> shift) & 1) {
      break;
    }
    float comp = values[local_id+stride];
    values[id] = MIN(values[id], comp)
    stride <<= 1;
    shift += 1;
  }
}


void LoadGlobalToLocalf(float *global, float *local, int width, uint id) {
  local[id] = (id < width) ? global[id] : INFINITY;
}

void LoadGlobalToLocali(uint __global *global, uint __local *local, int width, uint id) {
  local[id] = (id < width) ? global[id] : 0;
}



//Nearly the same thing as above, but with some extra fanciness
__kernel void UpdateVertex2(
__global uint *edge_source,
__global float *edge_weights,
__global float *distances,
__global uint *vertex_index, //where the edges for a given vertex start in the edges arrays
__global uint *num_edges //is this neccesary? should it be coputed with vertex_index
)
{
  uint start = get_group_id(0) * LOCAL_WORK_SIZE;
  uint local_id = get_local_id(0);
  float __local data[(LOCAL_WORK_SIZE << 3)];
  int rem_edges = num_edges[start];
  

  float *e_weights = data;
  float *s_weights = data + LWST2;
  float *updates = s_weights + LWST2;
  float *mins = updates + LWST2;
  uint to_load, i;

  for(i = 0; i < LOCAL_WORK_SIZE; i++) {
    uint item = start + i;
    float min = INFINITY;

    to_load = vertex_index[item];
    rem_edges = num_edges[item];
   
    while(rem_edges > 0) {
      uint source = edge_source[to_load + local_id];
      LoadGlobalToLocal(edge_weights + to_load, e_weights, rem_edges, local_id);
      s_weights[local_id] = distances[source];
      updates[local_id] = e_weights[local_id] + s_weights[local_id];
      rem_edges -= LOCAL_WORK_SIZE;
      to_load += LOCAL_WORK_SIZE;  //talk to jim about this?
      
      if(rem_edges <= 0) {
	barrier(CLK_LOCAL_MEM_FENCE); //This might not actually be necessary with LOCAL_WORK_SIZE = 32
	if(!(id & 1))
	   ComputeMin(updates, rem_edges + LOCAL_WORK_SIZE, local_id);
	if(min > updates[0])
	  min = updates[0];
      }
      
      source = edge_source[to_load + local_id];
      LoadGlobalToLocal(edge_weights + to_load, e_weights + LOCAL_WORK_SIZE, rem_edges, local_id);
      s_weights[local_id + LOCAL_WORK_SIZE] = distances[source];
      updates[local_id + LOCAL_WORK_SIZE] = e_weights[local_id] + s_weights[local_id];
      
      barrier(CLK_LOCAL_MEM_FENCE);
      
      ComputeMin(updates + offset, MIN(rem_edges + LOCAL_WORK_SIZE, LWST2), 
		 ((local_id & 1) ? (local_id + LOCAL_WORK_SIZE - 1) : local_id));

      rem_edges -= LOCAL_WORK_SIZE;
      to_load += LOCAL_WORK_SIZE;  //talk to jim about this?
    }
    
    mins[i] = min; //Only one thread needs to do this.  Maybe have thread 0 should be in charge?
    //In any case, I believe the desired behavior results
  }
  distances[start+local_id] = mins[local_id]; //Will be coalesced
}
*/

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
			   __global uint *num_edges //is this neccesary? should it be computed with vertex_index
)
{
  
  uint start = get_group_id(0) * LOCAL_WORK_SIZE;
  uint local_id = get_local_id(0);
  uint __local current_edge[LOCAL_WORK_SIZE];
  uint __local remaining_edges[LOCAL_WORK_SIZE];
  float __local e_weights[LOCAL_WORK_SIZE][HALF_WARP+1]; //since threads access this array colum
  uint __local  e_sources[LOCAL_WORK_SIZE][HALF_WARP+1];
  uint loading_id = local_id;
  int offset = 0;
  uint i;
  float min = distances[start+local_id];
  if(local_id >= HALF_WARP) {
    loading_id = local_id - (HALF_WARP);
    offset = 1;
  }
  current_edge[local_id] = vertex_index[get_global_id(0)];
  remaining_edges[local_id] = num_edges[get_global_id(0)];
  
  while(remaining_edges[0] > 0) {
    for(i = 0; i < LOCAL_WORK_SIZE; i += 2) {
      e_weights[i+offset][loading_id] = edge_weights[current_edge[i+offset]+loading_id];
      e_sources[i+offset][loading_id] = edge_source[current_edge[i+offset]+loading_id];
      /*
      LoadLocalToGlobalf((edge_weights + current_edge[i+offset]),
			 e_weights + (i+offset)*(HALF_WARP+1), 
			remaining_edges[i+offset],
			loading_id);
      LoadLocalToGlobali((edge_source + current_edge[i+offset]),
			 e_sources + (i+offset)*(HALF_WARP+1), 
			 remaining_edges[i+offset],
			 loading_id);
      */
    }
    uint max = MIN(HALF_WARP,remaining_edges[local_id]);
    for(i = 0; i < max; i++) {
    //possible optimization: see if the edge weight in question is,
    //by itself, smaller than the min value so far
    //if not, don't even bother with the global load from memory
      float temp = distances[e_sources[local_id][i]];
      if(temp < INFINITY) {
	temp = e_weights[local_id][i] + temp;
	min = min > temp ? temp : min;
      }
    }
    current_edge[local_id] += HALF_WARP;
    remaining_edges[local_id] -= HALF_WARP;
  }
  distances[start+local_id] = min;
}
