#include <iostream>
#include "triangle.hpp"
#define TOL 0.0001f
#define NON_ZERO(x) x < -TOL || x > TOL

using std::cout;
using std::endl;

bool Point::compare::operator()(const Point &q, const Point &p) const noexcept
{
	float diff = q.x - p.x;
	if(NON_ZERO(diff))
	{
		return diff < 0;
	}
	diff = q.y - p.y;
	if(NON_ZERO(diff))
	{
		return diff < 0;
	}
	diff = q.z - p.z;
	if(NON_ZERO(diff))
	{
		return diff < 0;
	}
	return false;
}

bool Point::operator==(const Point &p)
{
	if(NON_ZERO(x - p.x)) return false;
	if(NON_ZERO(y - p.y)) return false;
	if(NON_ZERO(z - p.z)) return false;
	return true;
}

Triangle::Triangle(float *p)
{
	for (int i = 0; i < 9; i++)
		points[i] = p[i];
	id = -1;
}

Triangle::~Triangle()
{

}

Point Triangle::getPoint(int i)
{
	return ((Point*)points)[i];
}

bool Triangle::isValid(float *p)
{
	float a[]={p[0]-p[3], p[1]-p[4], p[2]-p[5]};
	float b[]= {p[0]-p[6], p[1]-p[7], p[2]-p[8]};
	float x;

	x = a[0]*b[1] - b[0]*a[1];
	if (NON_ZERO(x)) return true;

	x = a[1]*b[2] - b[1]*a[2];
	if (NON_ZERO(x)) return true;

	x = a[0]*b[2] - b[0]*a[2];
	if (NON_ZERO(x)) return true;

	return false;
}

//helper function to swap the points if they are in the 'wrong' order
void _swap(float *p, float *q)
{
	if(p[0]>q[0]) return;
	if(p[0]==q[0])
	{
		if(p[1]>q[1]) return;
		if(p[1]==q[1])
		{
			if(p[2]>=q[2]) return;
		}
	}

	float temp;
	for(int i = 0; i < 3; i++)
	{
		temp = q[i];
		q[i] = p[i];
		p[i] = temp;
	}
}

void Triangle::sort(float *p)
{
	_swap(p, p+3);
	_swap(p, p+6);
	_swap(p+3, p+6);

}


bool Triangle::operator==(const Triangle &t)
{
	for (int i = 0; i < 9; i++)
	{
		if(NON_ZERO(points[i] - t.points[i]))
		{
			return false;
		}
	}
	return true;
}

bool Triangle::operator==(const float *p)
{
	for (int i = 0; i < 9; i++)
	{
		if(NON_ZERO(points[i] - p[i]))
		{
			return false;
		}
	}
	return true;
}


