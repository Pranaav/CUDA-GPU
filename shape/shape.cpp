#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include "shape.hpp"
#include <queue>

using std::cout;
using std::endl;
using std::vector;
using std::map;
using std::pair;


Shape::Shape()
{

}

Shape::~Shape()
{

}

bool Shape::_isNeighbour(uint t1, uint t2)
{
	int k = -1;
	for(int i=0; i<3; i++)
	{
		Point p = triangles.at(t1).getPoint(i);
		for(int j=0; j<3; j++)
		{
			if(k==j) continue;
			Point q = triangles.at(t2).getPoint(j);
			if(p==q)
			{
				if(k!=-1) return true;
				k=j;
				break;
			}
		}
	}
	return false;
}

bool Shape::addTriangle(float *p)
{
	if(!Triangle::isValid(p))
	{
		return false;
	}
	Triangle::sort(p);
	triangles.push_back(p);
	Triangle *t = &triangles.back();
	t->id = triangles.size() - 1;
	vector<uint> q;
	for(int i=0; i<3; i++)
	{
		auto it = pointmap.find(t->getPoint(i));
		if(it == pointmap.end())
		{
			vector<uint> v(1);
			v.push_back(t->id);
			pointmap.insert(pair<Point, vector<uint> >(t->getPoint(i), v));
		}
		else
		{
			for(uint j : it->second)
			{
				bool found = false;
				for(uint k : q)
				{
					if (k == j)
					{
						found = true;
						break;
					}
				}
				if(found) continue;
				if(_isNeighbour(j, t->id))
				{
					adjacencyLists.at(j).push_back(t->id);
					q.push_back(j);
				}
			}
			it->second.push_back(t->id);
		}
	}
	adjacencyLists.push_back(q);
	return true;
}

void Shape::status()//for debugging
{
	int N = triangles.size();
	cout << "status: " << endl;
	for(int i = 0; i < N; i++)
	{
		cout << i << ": ";
		for(auto t : adjacencyLists.at(i))
		{
			cout << t << " ";
		}
		cout << endl;
	}
}

bool Shape::isConnectedCpu(float *p, float *q)
{
	uint src,dest;
	if(!(findTriangle(p, src) && findTriangle(q, dest)))
	{
		return false;

	}
	if (src==dest) return true;

	std::queue<uint> que;
	uint n = adjacencyLists.size();
	bool *visit = new bool[n];
	for (int i = 0; i < n; ++i) {
		visit[i] = false;
	}
	visit[src] = true;
	que.push(src);
	while (!que.empty())
	{
		uint u = que.front();
		que.pop();
		for (uint v : adjacencyLists[u])
		{
			if (!visit[v])
			{
				if (v == dest) return true;
				que.push(v);
				visit[v] = true;
			}
		}
	}

	return visit[dest];
}

uint Shape::connectedComponentsCpu()
{

	int n = adjacencyLists.size();
	uint *color = new uint[n];
	for (int i = 0;i < n;i++) {
		color[i] = 0;
	}
	uint m = 0;
	for (int i = 0;i < n;i++) {
		if(color[i]==0)
		{
			m++;
			color_bfs(i, color, m);
		}
	}

	return m;
}

void Shape::color_bfs(uint src, uint *color, uint m)
{
	std::queue<uint> que;
	uint n = adjacencyLists.size();
	color[src] = m;
	que.push(src);
	while (!que.empty())
	{
		uint u = que.front();
		que.pop();
		for (uint v : adjacencyLists[u])
		{
			if (color[v] == 0)
			{
				que.push(v);
				color[v] = m;
			}
		}
	}
}


double Shape::getAvgDegree()
{
	size_t sum = 0;
	for(auto l : adjacencyLists)
	{
		sum+=l.size();
	}
	return ((double)sum) / adjacencyLists.size();
}
uint Shape::getSize()
{
	return triangles.size();
}
