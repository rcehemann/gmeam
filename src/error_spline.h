#ifndef ERROR_SPLINE_H
#define ERROR_SPLINE_H

#include <string>
#include <stdexcept>

namespace LAMMPS_NS {

class ErrorSpline : public std::runtime_error
{
public:
  ErrorSpline(const std::string& msg) : std::runtime_error(msg) {}
}; // ErrorSpline

} // LAMMPS_NS

#endif // ERROR_SPLINE_H

