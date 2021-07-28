#include <cstddef>
#include <cstdlib>
#include <iostream>
#include "shape.hpp"


using std::cout;
using std::endl;
using std::vector;
using std::map;
using std::pair;

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)


void Shape::initForBFS(uint* &nov, uint* &adj)
{
    uint *h_nov, *h_adj;
    uint total = 0;
    const uint numv = triangles.size();
    for(auto i : adjacencyLists)
	{
		total += i.size();
	}
    const uint sizeV = (numv+1) * sizeof(uint),
                    sizeA = total * sizeof(uint);

    CUDA_CHECK_RETURN(cudaMallocHost(&h_nov, sizeV));
    CUDA_CHECK_RETURN(cudaMallocHost(&h_adj, sizeA));
    h_nov[0] = 0;

    uint k = 0;

    for(uint i = 0; i < numv; i++)
    {
        h_nov[i] = k;
        const uint size = adjacencyLists.at(i).size();
        for(uint j = 0; j < size; j++)
        {
            h_adj[k] = adjacencyLists.at(i).at(j);
            k++;
        }
    }
    h_nov[numv] = k;


    CUDA_CHECK_RETURN(cudaMalloc(&nov, sizeV));
    CUDA_CHECK_RETURN(cudaMalloc(&adj, sizeA));
    CUDA_CHECK_RETURN(cudaMemcpy(nov, h_nov, sizeV, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(adj, h_adj, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaFreeHost(h_nov));
    CUDA_CHECK_RETURN(cudaFreeHost(h_adj));
}

bool Shape::findTriangle(float *p, uint &id)
{
	auto iter = pointmap.find(arrToPoint(p));
    if(iter == pointmap.end())
    {
        return false;

    }
	Triangle::sort(p);
    for(uint i : iter->second)
    {
        if(triangles.at(i)==p){
            id = i;
            return true;
        }
    }
    return false;
}
__global__
void _init1(bool *f, bool *s, bool *flag, const int u, const uint numv)
{
	uint i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numv)
	{
		if (i == u)
		{
			f[i] = true;
			s[i] = true;
			*flag = false;
		}
		else
		{
			f[i] = false;
			s[i] = false;
		}
	}
}

__global__
void connected_kernel(const uint *nov, const uint *adj, const int dest, const int numv, bool *f, bool *s, bool *flag)
{
    uint i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numv && f[i])
    {
    	f[i] = false;
        uint v = nov[i];
        uint u = nov[i+1];
        for(uint j = v; j < u; j++)
        {
            int k = adj[j];
            if(!s[k])
            {
                f[k] = true;
                s[k] = true;
				flag[0] = false;
                if(k==dest)
                {
                	flag[1] = true;
                }
            }
        }
    }
}

bool Shape::isConnectedGpu(float *p, float *q)
{
    uint src,dest;
    if(!(findTriangle(p, src) && findTriangle(q, dest)))
    {
        return false;
    }
    if (src==dest) return true;
    uint *nov, *adj;
    initForBFS(nov, adj);

    bool *f, *s;
    bool *flag;
    const uint numv = triangles.size();
    uint size = numv * sizeof(bool);
    CUDA_CHECK_RETURN(cudaMalloc(&f, size));
    CUDA_CHECK_RETURN(cudaMalloc(&s, size));
    CUDA_CHECK_RETURN(cudaMalloc(&flag, 2*sizeof(bool)));
    int tpb = 128;
    int bpg =(numv + tpb - 1) / tpb;

    _init1<<<bpg,tpb>>>(f, s, flag + 1, src, numv);
    CUDA_CHECK_RETURN(cudaGetLastError());
    bool m[2]  = {true, false};
    while(true)
    {
    	CUDA_CHECK_RETURN(cudaMemset(flag, true, sizeof(bool)));
        connected_kernel<<<bpg,tpb>>>(nov, adj, dest, numv, f, s, flag);
        CUDA_CHECK_RETURN(cudaGetLastError());
        CUDA_CHECK_RETURN(cudaMemcpy(m, flag, 2*sizeof(bool), cudaMemcpyDeviceToHost));
        if(m[0] || m[1]) break;
    }

    CUDA_CHECK_RETURN(cudaFree(flag));
    CUDA_CHECK_RETURN(cudaFree(nov));
    CUDA_CHECK_RETURN(cudaFree(adj));
    CUDA_CHECK_RETURN(cudaFree(f));
    CUDA_CHECK_RETURN(cudaFree(s));
    return m[1];
}

__global__
void _init2(bool *f, bool *nf, uint *col, const uint numv)
{
	uint i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numv)
	{
		f[i] = true;
		nf[i] = false;
		col[i] = i;
	}
}

__global__
void colorcomponentsgpu(const uint* nov, const uint* adj, bool* f, bool* nf, uint* color,const uint numv, bool *flag)
{
	uint i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numv)
	{
		if (f[i])
		{
			f[i] = false;
			uint hedcol = color[i];
			const uint ci = nov[i];
			const uint cj = nov[i + 1];
			bool ert = false;
			for (uint l = ci;l < cj;l++)
			{
				uint lefcol = color[adj[l]];
				if (lefcol > hedcol)
				{
					atomicMin(color + adj[l], hedcol);
					nf[adj[l]] = true;
					flag[0] = true;
				}
				else if (hedcol > lefcol)
				{
					ert = true;
					hedcol = lefcol;
				}
			}
			if (ert)
			{
				atomicMin(color + i, hedcol);
				nf[i] = true;
				flag[0] = true;
			}
		}
	}

}

__global__
void countcol(uint *col, uint* count, const uint numv)
{
	uint i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numv)
	{
		if(col[i]==i)
		{
			atomicAdd((uint*)count, 1ull);
		}
	}
}

uint Shape::connectedComponentsGpu()
{
    uint *nov, *adj, *col;
    bool *f, *nf;
	bool m = true;
	bool *flag;
	const uint numv = triangles.size();
	const uint size = numv * sizeof(bool);
    initForBFS(nov, adj);
	CUDA_CHECK_RETURN(cudaMalloc(&f, size));
	CUDA_CHECK_RETURN(cudaMalloc(&nf, size));
	CUDA_CHECK_RETURN(cudaMalloc(&col, numv * sizeof(uint)));
	CUDA_CHECK_RETURN(cudaMalloc(&flag, sizeof(bool)));
	int tpb = 128;
	int bpg = (numv + tpb - 1) / tpb;
	_init2<<<bpg,tpb>>>(f,nf,col,numv);
	while(m)
	{
    	CUDA_CHECK_RETURN(cudaMemset(flag, false, sizeof(bool)));
    	colorcomponentsgpu<<<bpg,tpb>>>(nov, adj, f, nf, col, numv, flag);
    	CUDA_CHECK_RETURN(cudaGetLastError());
    	CUDA_CHECK_RETURN(cudaMemcpy(&m, flag, sizeof(bool), cudaMemcpyDeviceToHost));
    	bool *tmp = nf;
    	nf = f;
    	f = tmp;
	}
	uint ret;
	uint* count;
	CUDA_CHECK_RETURN(cudaMalloc(&count, sizeof(uint)));
	countcol<<<bpg,tpb>>>(col, count, numv);
	CUDA_CHECK_RETURN(cudaMemcpy(&ret, count, sizeof(uint), cudaMemcpyDeviceToHost));


    CUDA_CHECK_RETURN(cudaFree(flag));
    CUDA_CHECK_RETURN(cudaFree(count));
    CUDA_CHECK_RETURN(cudaFree(nov));
    CUDA_CHECK_RETURN(cudaFree(adj));
    CUDA_CHECK_RETURN(cudaFree(f));
    CUDA_CHECK_RETURN(cudaFree(nf));
    CUDA_CHECK_RETURN(cudaFree(col));
	return ret;
}

static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (-1);
}
