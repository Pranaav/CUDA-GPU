#ifndef SHAPE_HPP
#define SHAPE_HPP

#include <fstream>
#include <vector>
#include "triangle.hpp"
#include <map>


using std::vector;
using std::map;
using std::pair;


class Shape
{
private:
    map<Point,vector<uint>,Point::compare> pointmap;
    vector<Triangle> triangles;
    vector<vector<uint>> adjacencyLists;
    void initForBFS(uint* &V, uint* &Adj);
    bool _isNeighbour(uint t1, uint t2);
    void color_bfs(uint src, uint *color, uint m);
public:
    Shape();
    ~Shape();
    bool addTriangle(float *p);
    bool findTriangle(float *p, uint &id);
    bool isConnectedGpu(float *p, float *q);
    bool isConnectedCpu(float *p, float *q);
    uint connectedComponentsGpu();
    uint connectedComponentsCpu();
    void status();
    double getAvgDegree();
    uint getSize();

};


#endif // SHAPE_HPP
