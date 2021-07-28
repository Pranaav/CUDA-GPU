#ifndef TRIANGLE_HPP
#define TRIANGLE_HPP

#include <cstddef>

typedef uint32_t uint;


struct Point{
    float x;
    float y;
    float z;
    bool operator==(const Point &p);
    struct compare
	{
		bool operator()(const Point &p, const Point &q) const noexcept;
	};
};
#define arrToPoint(p) *(Point*)p

class Triangle
{
private:
    float points[9];
public:
    uint id;
    Triangle(float *p);
    ~Triangle();

    Point getPoint(int);
    static void sort(float *p);
    static bool isValid(float *p);
    bool operator==(const Triangle &t);
    bool operator==(const float *p);

};


#endif // TRIANGLE_HPP
