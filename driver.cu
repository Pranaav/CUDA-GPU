
#include <iostream>
#include <fstream>
#include <cstring>
#include "shape/shape.hpp"
#include <chrono>

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

using std::cout;
using std::endl;
using std::vector;
using namespace std::chrono;

typedef uint32_t uint;

int main(int argc, char *argv[])
{

	Shape s;
	vector<bool> acc[2];
	vector<size_t>  time[2][2];
	std::ifstream in;
	std::ofstream out;
	if(argc > 1)
	{
		out.open(argv[2]);
	}
	else
	{
		out.open("src/out");

	}
	if(argc > 0)
	{
		in.open(argv[1]);
	}
	else
	{
		in.open("src/inp");
	}
	char line[50];
	CUDA_CHECK_RETURN(cudaSetDevice(0));
	while(in >> line)
	{

		if(std::strcmp(line, "ADD_TRIANGLE")==0)
		{
			float p[9];
			for (int i = 0; i < 9; ++i)
			{
				in >> p[i];
			}
			s.addTriangle(p);
		}
		else if(std::strcmp(line, "STATUS")==0){
			s.status();
		}
		else if(std::strcmp(line, "IS_CONNECTED")==0)
		{
			float p[9], q[9];
			for (int i = 0; i < 9; ++i)
			{
				in >> p[i];
			}
			for (int i = 0; i < 9; ++i)
			{
				in >> q[i];
			}
			auto t1 = high_resolution_clock::now();
			bool retGpu = s.isConnectedGpu(p, q);
			auto t2 = high_resolution_clock::now();
			bool retCpu = s.isConnectedCpu(p, q);
			auto t3 = high_resolution_clock::now();
			time[0][0].push_back(duration_cast<microseconds>(t2-t1).count());
			time[0][1].push_back(duration_cast<microseconds>(t3-t2).count());
			acc[0].push_back(retCpu==retGpu);
			if(retCpu == retGpu)
			{
				cout << (retGpu?"connected":"not connected") << endl;
			}
			else
			{
				cout << "wrong " << retGpu << " " << retCpu << endl;
			}
		}
		else if(std::strcmp(line, "FIND_TRIANGLE")==0)
		{
			float p[9];
			for (int i = 0; i < 9; ++i)
			{
				in >> p[i];
			}
			uint id;
			bool ret = s.findTriangle(p, id);
			if(ret)
				cout << "found " << id << endl;
			else
				cout << "not found" << endl;
		}
		else if(std::strcmp(line, "COUNT_CONNECTED_COMPONENTS")==0)
		{
			auto t1 = high_resolution_clock::now();
			uint retGpu = s.connectedComponentsGpu();
			auto t2 = high_resolution_clock::now();
			uint retCpu = s.connectedComponentsCpu();
			auto t3 = high_resolution_clock::now();
			time[1][0].push_back(duration_cast<microseconds>(t2-t1).count());
			time[1][1].push_back(duration_cast<microseconds>(t3-t2).count());
			acc[1].push_back(retCpu==retGpu);

			if(retCpu == retGpu)
			{
				cout << retGpu << endl;
			}
			else
			{
				cout << "wrong " << retGpu << " " << retCpu << endl;
			}
		}
		else if(std::strcmp(line, "TIME")==0)
		{
			out << "# of triangles = " << s.getSize() << endl;
			out << "average degree = " << s.getAvgDegree() << endl;
			const char* s[] = {"isConnected","NComponents"};
			for(int i = 0; i < 2; i++)
			{
				int n  = acc[i].size();
				for(int j = 0; j<n; j++)
				{
					out << s[i] << " " << acc[i].at(j) << " " << time[i][0].at(j) << " " << time[i][1].at(j) << endl;
				}
			}
		}

	}
	out.close();
	in.close();
	return 0;
}

static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

