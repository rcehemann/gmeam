#ifndef SPLINE_H
#define SPLINE_H

#include "pointers.h"

#include <vector>
#include <istream>

namespace LAMMPS_NS {

class Spline : protected Pointers
{
public:
  Spline(class LAMMPS *);
  virtual ~Spline();

  double get_cutoff() const;                                    // Get radial cutoff of spline (basically last knot)
  virtual double splint_comb(double, double&) const;            // Interpolate fn & grad from splines given knot_idx
  virtual void communicate();                                   // Broadcasts the spline function parameters to all processors

  friend std::istream& operator>>(std::istream&, Spline&);      // Read in spline from input stream
  Spline& operator=(const Spline& rhs);

protected:
  int nknots_;
  std::vector<double> x_, y_, ypp_;
  double yp0_, ypn_, xmax_shifted_;

  // Edit later for equidistant/non-equidistant splines
  double step_, invstep_;

  virtual void read(std::istream&);   // Read in spline data from input stream
  virtual void resize();              // Resize spline for # knots
  void rehash();                      // Setup spline using limited data already possessed

private:

}; // Spline

/* ----------------------------------------------------------------------
   Interpolates function and gradient with equidistant splines
------------------------------------------------------------------------- */

inline
double Spline::splint_comb(double r, double& grad) const
{
  double delta_r = r - x_[0];

  if (delta_r > 0 && delta_r < xmax_shifted_) {
    int k = int(delta_r * invstep_);
    if ( k >= nknots_-1 ) k = nknots_-2;  // Stay within one less than spline upper bound
    double b = (delta_r - k*step_)*invstep_;
    double a = 1.0 - b;
    double p1 = y_[k];
    double d21 = ypp_[k];
    double p2 = y_[++k];
    double d22 = ypp_[k];

    grad = (p2-p1)*invstep_ + ((3*b*b-1)*d22 - (3*a*a-1)*d21)*step_/6.0;
    return a*p1 + b*p2 + ((a*a*a-a)*d21 + (b*b*b-b)*d22)*step_*step_/6.0;
  } else if (delta_r <= 0) {
    grad = yp0_;
    return y_[0] + yp0_ * delta_r;
  } else {
    grad = ypn_;
    return y_[nknots_-1] + ypn_ * (r - x_[nknots_-1]);
  }
}

} // LAMMPS_NS

#endif // SPLINE_H
