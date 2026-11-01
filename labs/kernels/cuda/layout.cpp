#include <array>
#include <cassert>
#include <iostream>
#include <vector>
int main() {
    constexpr int rows=3, cols=4;
    std::vector<float> a(rows*cols);
    for (int i=0;i<rows*cols;++i) a[i]=static_cast<float>(i);
    const auto at=[&](int row,int col){return a.at(row*cols+col);};
    const auto transposed=[&](int row,int col){return a.at(col*cols+row);};
    assert(at(1,2)==6); assert(transposed(2,1)==6);
    std::cout << "offset(1,2)=" << 1*cols+2 << " value=" << at(1,2)
              << " transpose(2,1)=" << transposed(2,1) << '\n';
    // data() points to storage; a C++ pointer carries no tensor shape.
    const float* p=a.data();
    std::cout << "sizeof(float)=" << sizeof(float) << " p[6]=" << p[6] << '\n';
}
