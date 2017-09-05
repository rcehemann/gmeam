#ifdef PAIR_CLASS

PairStyle(gmeam/spline,PairGMEAMSpline)

#else

#ifndef LMP_PAIR_GMEAM_SPLINE_H
#define LMP_PAIR_GMEAM_SPLINE_H

#include "pair.h"
#include "spline.h"

#include <vector>
#include <map>
#include <iostream>

namespace LAMMPS_NS {

class PairGMEAMSpline : public Pair
{
public:
  PairGMEAMSpline(class LAMMPS *);
  virtual ~PairGMEAMSpline();

  virtual void compute(int, int);
  void allocate();
  void settings(int, char **);
  void coeff(int, char **);

//  int pack_forward_comm(int, int *, double *, int, int *);
//  void unpack_forward_comm(int, int, double *);
//  int pack_reverse_comm(int, int, double *);
//  void unpack_reverse_comm(int, int *, double *);
//  double memory_usage();

  void init_style();
  void init_list( int, class NeighList *);
  double init_one(int, int);
 // double* Uprime_values; // stores values of U'(rho)
  int nmax;	// size of temporary array for Uprime
protected:

  std::vector<Spline> phi; 
  std::vector<Spline> rho;
  std::vector<Spline> u;
  std::vector<Spline> f;
  std::vector<Spline> g;

  double cutoff_max;

  virtual void read_file(std::string);    // Read in potential from file

private:

  struct MEAM2Body {
    int idx, typ;
    double r, recip, nx, ny, nz, f, df;
  };
  
  struct MEAM3Body {
    double cos, g, dg;
  };

//  double* zero_atom_energies;
  std::vector<MEAM2Body> two_body_pair_info;
  std::vector<MEAM2Body> two_body_trip_info;
  std::vector<MEAM3Body> three_body_info;

  std::vector<std::string> map_atom_type;
  std::map<std::string, int> elements;
//  class NeighList *list;
};  // PairGMEAMSpline

} // LAMMPS_NS

#endif  // LMP_PAIR_MEAM_ALLOY_SPLINE_H
#endif  // PAIR_CLASS
