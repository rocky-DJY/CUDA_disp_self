//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: corre.cpp
// MATLAB Coder version            : 4.3
// C/C++ source code generated on  : 28-Sep-2020 09:02:38
// Include Files
#include "corre.h"
#include <cstring>

// Function Definitions

//
// Arguments    : const float x[dim]
//                const float y[dim]
//                float res[101]
//                float lag[101]
// Return Type  : void
//
void corre(const float *x, const float *y, float *res, float *lag,const int dim)
//   x   [51]
//   y   [51]
//   res [101]
//   lag [101]
//   dim 51
{
  int res_len=dim*2-1;    //101
  int m_size=dim-1;       //50
  int n_size=dim-2;       //49
  int k;
  int ihi;
  float s;
  int i;
  std::memset(&res[0], 0, res_len * sizeof(float));
  for (k = 0; k < dim; k++) {
    if (dim - k < dim) {
      ihi = m_size - k;
    } else {
      ihi = m_size;
    }
    s = 0.0;
    for (i = 0; i <= ihi; i++) {
      s += y[i] * x[k + i];
    }
    res[k + m_size] = s;
  }
  for (k = 0; k < m_size; k++) {
    ihi = n_size - k;
    s = 0.0;
    for (i = 0; i <= ihi; i++) {
      s += y[(k + i) + 1] * x[i];
    }
    res[n_size - k] = s;
  }
  for (k = 0; k < res_len; k++) {
    lag[k] = static_cast<float>(k) + -m_size;
  }
}

//
// File trailer for corre.cpp
//
// [EOF]
//
