#ifndef SPLINE5_H
#define SPLINE5_H

#include "spline.h"

namespace LAMMPS_NS {

class Spline5 : public Spline
{
public:
  Spline5(class LAMMPS *);
  virtual ~Spline5();

  virtual double splint_comb(double, double&) const;            // Interpolate fn & grad from splines given knot_idx
  virtual void communicate();                                   // Broadcasts the spline function parameters to all processors
  Spline5& operator=(const Spline5& rhs);
protected:
  std::vector<double> yp4_;

  virtual void read(std::istream&);   // Read in spline data from input stream
  virtual void resize();              // Resize spline for # knots

private:

}; // Spline5

/* ----------------------------------------------------------------------
   Interpolates function and gradient with equidistant splines
------------------------------------------------------------------------- */

inline
double Spline5::splint_comb(double r, double& grad) const
{
  double delta_r = r - x_[0];

  if (delta_r > 0 && delta_r < xmax_shifted_) {
    int k = int(delta_r * invstep_);
    if ( k >= nknots_-1 ) k = nknots_-2;  // Stay within one less than spline upper bound
    double b = (delta_r - k*step_)*invstep_;
    double a = 1.0 - b;
    double p1 = y_[k];
    double d21 = ypp_[k];
    double d41 = yp4_[k];
    double p2 = y_[++k];
    double d22 = ypp_[k];
    double d42 = yp4_[k];

    grad = (p2-p1)*invstep_ + ((3*b*b-1)*d22 - (3*a*a-1)*d21)*step_/6.0 +
           ((15*b*b*b*b-30*b*b+7)*d42 - (15*a*a*a*a-30*a*a+7)*d41)*(step_*step_*step_)/360.0;
    return a*p1 + b*p2 + ((a*a*a-a)*d21 + (b*b*b-b)*d22)*(step_*step_)/6.0 +
           ((3*a*a*a*a*a-10*a*a*a+7*a)*d41 + (3*b*b*b*b*b-10*b*b*b+7*b)*d42)*(step_*step_*step_*step_)/360.0;
  } else if (delta_r <= 0) {
    grad = yp0_ + ypp_[0] * delta_r;
    return y_[0] + yp0_ * delta_r + 0.5 * ypp_[0] * delta_r * delta_r;
  } else {
    int kn = nknots_-1;
    delta_r = (r - x_[kn]);  // displacement from right most knot
    grad = ypn_ + ypp_[kn] * delta_r;
    return y_[kn] + ypn_ * delta_r + 0.5 * ypp_[kn] * delta_r * delta_r;
  }
}

} // LAMMPS_NS

#endif // SPLINE5_H
