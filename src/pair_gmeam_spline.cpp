#include "pair_gmeam_spline.h"
#include "atom.h"
#include "memory.h"
#include "error.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "force.h"
#include "comm.h"
#include "error_spline.h"

#include <sstream>
#include <fstream>
#include <cmath>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairGMEAMSpline::PairGMEAMSpline(LAMMPS *lmp) : Pair(lmp), cutoff_max(0.0)
{

  single_enable = 0;        // 1 if single() routine exists
  restartinfo = 0;          // 1 if pair style writes restart info
  one_coeff = 1;            // 1 if allows only one coeff * * call

  nmax = 0;

  comm_forward = 0;         // size of forward communication (0 if none)
  comm_reverse = 0;         // size of reverse communication (0 if none)
}

/* ---------------------------------------------------------------------- */

PairGMEAMSpline::~PairGMEAMSpline()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
   // memory->destroy(Uprime_values);

//    delete[] zero_atom_energies;

  }
}

/* ----------------------------------------------------------------------
   Compute energy, stresses, and forces
------------------------------------------------------------------------- */

void PairGMEAMSpline::compute(int eflag, int vflag)
{

  if (eflag || vflag) ev_setup(eflag, vflag);
  else evflag = vflag_fdotr = eflag_global = vflag_global = eflag_atom = vflag_atom = 0;

  double cutoff_maxsq = cutoff_max * cutoff_max;

  double** const x = atom->x;
  double **F = atom->f;
  int *type = atom->type;
  int ntypes = atom->ntypes;
  int nlocal = atom->nlocal;
  bool newton_pair = force->newton_pair;

  // using full atom list
  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

//  // grow per-atom array if necessary
//  if (atom->nmax > nmax) {
//	memory->destroy(Uprime_values);
//	nmax = atom->nmax;
//	memory->create(Uprime_values, nmax, "pair:Uprime");
//  }

  // Determine the maximum number of neighbors a single atom has
  int newMaxNeighbors = 0;
  for(int ii = 0; ii < inum; ++ii) {
    int jnum = numneigh[ilist[ii]];
    if(jnum > newMaxNeighbors) newMaxNeighbors = jnum;
  }

  // Allocate array for temporary bond info
  int maxNeighbors = two_body_pair_info.size();
  if(newMaxNeighbors > maxNeighbors) {
    two_body_pair_info.resize(newMaxNeighbors);
    two_body_trip_info.resize(newMaxNeighbors);
  }

  // calculate max number of triplets for any atom
  maxNeighbors = two_body_trip_info.size();
  int newMaxTriplets = (maxNeighbors * (maxNeighbors - 1))/2;

  // allocate array for temporary triplet info
  int maxTriplets = three_body_info.size();
  if (newMaxTriplets > maxTriplets) {
     three_body_info.resize(newMaxTriplets);
  }
  


  // Loop over full neighbor list of my atoms
  for(int ii = 0; ii < inum; ++ii) {
    int i = ilist[ii];
    double xtmp = x[i][0];
    double ytmp = x[i][1];
    double ztmp = x[i][2];
    int itype = type[i]-1;      // shifted back for reference to arrays of Splines
    double rhovalue = 0.0;

    // Two-body interactions
    int *jlist = firstneigh[i];
    int jnum = numneigh[i];
    int n2 = 0; // keep count of 2-body EAM  info for atom ii
    int n3 = 0; // keep count of 2-body MEAM info for atom ii

    for (int jj = 0; jj < jnum; ++jj) {    //loop over neighbors
      int j = jlist[jj];
      j = j & NEIGHMASK;

      int jtype = type[j]-1;      // species of current neighbor. shifted back for reference to arrays
      int pair_idx = itype*ntypes + jtype;
      
      double delx = x[j][0] - xtmp;
      double dely = x[j][1] - ytmp;
      double delz = x[j][2] - ztmp;
      double rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cutoff_maxsq) { 

        double r   = std::sqrt(rsq);
	double partial_sum = 0.0;
        double recip = 1.0/r;

	// first pair potential terms
	if (r < phi[pair_idx].get_cutoff()) {

          // Compute phi(r_ij) and its gradient in one step
          // Only half of the gradient contributes to the force as
          // well as half of the energy since we are double counting
          double phigrad;
          double phival = 0.5 * phi[pair_idx].splint_comb(r, phigrad);
     	  double tmp_force = 0.5 * phigrad * recip;
          
	  // Add in force on atom i from atom j
          // Subtract off force on atom j from atom i
          // Newton's law: action = -reaction
     	  F[i][0] += tmp_force * delx;
     	  F[i][1] += tmp_force * dely;
     	  F[i][2] += tmp_force * delz;
     	  F[j][0] -= tmp_force * delx;
     	  F[j][1] -= tmp_force * dely;
     	  F[j][2] -= tmp_force * delz;
          
	  if (evflag) ev_tally(i,j,nlocal,newton_pair,phival,0.0,-tmp_force,delx,dely,delz);
	}// finished with pair potential terms

	// now store pair density info for later use (EAM component)
	if (r < rho[jtype].get_cutoff()) {
	  
	  rhovalue += rho[jtype].splint_comb(r, two_body_pair_info[n2].df);
	 
          two_body_pair_info[n2].idx = j;
          two_body_pair_info[n2].r = r;
          two_body_pair_info[n2].recip = recip;
          two_body_pair_info[n2].nx = delx*recip;
          two_body_pair_info[n2].ny = dely*recip;
          two_body_pair_info[n2].nz = delz*recip;

	  ++n2; // increment number of EAM-component two-body terms
	}

	// now store triplet two-body info (MEAM component)
	if (r < f[pair_idx].get_cutoff()) {

	  two_body_trip_info[n3].f = f[pair_idx].splint_comb(r, two_body_trip_info[n3].df);

          two_body_trip_info[n3].idx = j;
          two_body_trip_info[n3].typ = jtype;
          two_body_trip_info[n3].r = r;
          two_body_trip_info[n3].recip = recip;
          two_body_trip_info[n3].nx = delx*recip;
          two_body_trip_info[n3].ny = dely*recip;
          two_body_trip_info[n3].nz = delz*recip;


	  ++n3; // increment number of MEAM-component two-body terms
	}
      }	// rsq < cutoff_maxsq
    } // END LOOP OVER NEIGHBORS jj
	  
    // calculate angular contribution to density
    int nijk = 0;
    for (int jj = 0; jj < n3-1; ++jj) {
       const MEAM2Body& neigh_jj = two_body_trip_info[jj];
	
       for (int kk = jj+1; kk < n3; ++kk) {
	  const MEAM2Body& neigh_kk = two_body_trip_info[kk];

	  // calculate theta_jik with dot product 
          double cos_theta = (neigh_jj.nx * neigh_kk.nx +
                	      neigh_jj.ny * neigh_kk.ny +
                	      neigh_jj.nz * neigh_kk.nz);

	  int jtype = neigh_jj.typ;
	  int ktype = neigh_kk.typ;
	  
	  int trip_idx = itype*ntypes*ntypes + jtype*ntypes + ktype;

	  three_body_info[nijk].cos = cos_theta;
	  three_body_info[nijk].g   = g[trip_idx].splint_comb(cos_theta, three_body_info[nijk].dg);
	  rhovalue 		   += neigh_jj.f * neigh_kk.f * three_body_info[nijk].g;
         
	  ++nijk; 
       }  // END LOOP OVER NEIGHBORS KK
    } // END LOOP OVER NEIGHBORS JJ

    // Done with calculating rho[i], now we can calculate the embedding energy
    double du;
    double junk;
    double embedding_energy = u[itype].splint_comb(rhovalue, du);// - u[itype].splint_comb(0.0, junk);
    if(eflag) { //embedding energy gets added to global and per/atom energies, not tallied with pairwise energy
      if(eflag_global) eng_vdwl += embedding_energy;
      if(eflag_atom) eatom[i] += embedding_energy;
    }

    // forces from embedding potential and pair-density term
    for (int jj = 0; jj<n2; ++jj) {
      
      const MEAM2Body& neigh_jj = two_body_pair_info[jj];
      int j = neigh_jj.idx;
      double tmp_force = neigh_jj.df * du;
      
      F[i][0] += tmp_force * neigh_jj.nx;
      F[i][1] += tmp_force * neigh_jj.ny;
      F[i][2] += tmp_force * neigh_jj.nz;

      F[j][0] -= tmp_force * neigh_jj.nx;
      F[j][1] -= tmp_force * neigh_jj.ny;
      F[j][2] -= tmp_force * neigh_jj.nz;
      if (vflag_either) {
        double delx = neigh_jj.nx * neigh_jj.r;
        double dely = neigh_jj.ny * neigh_jj.r;
        double delz = neigh_jj.nz * neigh_jj.r;
        ev_tally(i,j,nlocal,newton_pair,0.0,0.0,-tmp_force*neigh_jj.recip,delx,dely,delz);
      }
    } // done with pair-desntiy forces

    // three-body terms
    nijk = 0;
    for (int jj = 0; jj < n3-1; ++jj) {

      const MEAM2Body& neigh_jj = two_body_trip_info[jj];
      int j = neigh_jj.idx;
      int jtype = type[j]-1;      // species of current neighbor. shifted back for reference to arrays

      for (int kk = jj+1; kk < n3; ++kk) {
        const MEAM2Body& neigh_kk = two_body_trip_info[kk];
	const MEAM3Body& triplet_ijk = three_body_info[nijk];

	int k = neigh_kk.idx;
	double rik = neigh_kk.r;

	double dV3j = triplet_ijk.g  * neigh_jj.df * neigh_kk.f  * du;
	double dV3k = triplet_ijk.g  * neigh_jj.f  * neigh_kk.df * du;
	double V3   = triplet_ijk.dg * neigh_jj.f  * neigh_kk.f  * du;

	double vlj  = V3 * neigh_jj.recip;
	double vlk  = V3 * neigh_kk.recip;
	double vv3j = dV3j - vlj * triplet_ijk.cos;
	double vv3k = dV3k - vlk * triplet_ijk.cos;
	
	double fj[3], fk[3];

	fj[0] = neigh_jj.nx * vv3j + neigh_kk.nx * vlj; 
	fj[1] = neigh_jj.ny * vv3j + neigh_kk.ny * vlj; 
	fj[2] = neigh_jj.nz * vv3j + neigh_kk.nz * vlj;

	fk[0] = neigh_kk.nx * vv3k + neigh_jj.nx * vlk; 
	fk[1] = neigh_kk.ny * vv3k + neigh_jj.ny * vlk; 
	fk[2] = neigh_kk.nz * vv3k + neigh_jj.nz * vlk;

	// force on atom i	
	F[i][0] += fj[0] + fk[0];
	F[i][1] += fj[1] + fk[1];
	F[i][2] += fj[2] + fk[2];

	// reactive force on j 
	F[j][0] -= fj[0];
	F[j][1] -= fj[1];
	F[j][2] -= fj[2];

	// reactive force on k
	F[k][0] -= fk[0];
	F[k][1] -= fk[1];
	F[k][2] -= fk[2];

	if(evflag) {
	  double delta_ij[3]; double fjv[3];
	  double delta_ik[3]; double fkv[3];
	  
          delta_ij[0] = neigh_jj.nx * neigh_jj.r;
          delta_ij[1] = neigh_jj.ny * neigh_jj.r;
          delta_ij[2] = neigh_jj.nz * neigh_jj.r;
          delta_ik[0] = neigh_kk.nx * neigh_kk.r;
          delta_ik[1] = neigh_kk.ny * neigh_kk.r;
          delta_ik[2] = neigh_kk.nz * neigh_kk.r;

	  fjv[0] = -fj[0]; fkv[0] = -fk[0];
	  fjv[1] = -fj[1]; fkv[1] = -fk[1];
	  fjv[2] = -fj[2]; fkv[2] = -fk[2];

          ev_tally3(i, j, k, 0.0, 0.0, fjv, fkv, delta_ij, delta_ik);
	}

	++nijk;
      } // LOOP OVER NEIGHBORS KK
    } // LOOP OVER NEIGHBORS JJ
  } // LOOP OVER ATOMS II

  if(vflag_fdotr) virial_fdotr_compute();

  return;
}

/* ---------------------------------------------------------------------- */

void PairGMEAMSpline::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  
  //zero_atom_energies = new double[n];
}

/* ----------------------------------------------------------------------
   Global settings
------------------------------------------------------------------------- */

void PairGMEAMSpline::settings(int narg, char **arg)
{
  if(narg != 0) error->all(FLERR,"Illegal pair_style command");
  return;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairGMEAMSpline::coeff(int narg, char **arg)
{
  std::vector<std::string> args(narg);
  for (int i = 0; i < narg; ++i) args[i] = arg[i];

  if (!allocated) allocate();

  if (narg != 3 + atom->ntypes)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // ensure I,J args are * *
  if (args[0] != "*" || args[1] != "*")
    error->all(FLERR,"Incorrect args for pair coefficients");

  // WARNING: This is still unnecessary
  
  // read args that map atom types to elements in potential file
  // map[i] = which element the Ith atom type is, -1 if NULL
  // nelements = # of unique elements
  // elements = list of element names

  map_atom_type = std::vector<std::string>(atom->ntypes, "");
  elements = std::map<std::string, int>();

  int nelements = 0;
  for (int i = 3; i < narg; ++i) {
    std::string element_name = args[i];

    // atom_type -> element_name
    map_atom_type[i-3] = element_name;

    if (element_name == "NULL") {
      elements[element_name] = -1;
      continue;
    }

    // element_name -> potential_idx
    std::map<std::string, int>::iterator it;
    it = elements.find(element_name);
    if (it == elements.end()) { // not found, count this as new element
      elements[element_name] = nelements;
      ++nelements;
    }
  }
  
  // read potential file
  read_file(args[2]);

  // FROM PREVIOUS WARNING
  
   //clear setflag since coeff() called once with I,J = * *
  int n = atom->ntypes;
  for (int i = 1; i <= n; ++i) {
    for (int j = i; j <= n; ++j) {
      setflag[i][j] = 0;
    }
  }

  // set setflag i,j for type pairs where both are mapped to elements
  int count = 0;
  for (int i = 1; i <= n; ++i) {
    for (int j = i; j <= n; ++j) {
      setflag[i][j] = 1;
      ++count;
    }
  }
  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
  
}

/* ----------------------------------------------------------------------
   Init specific to this pair style
------------------------------------------------------------------------- */

void PairGMEAMSpline::init_style()
{

  if(force->newton_pair == 0)
    error->all(FLERR,"Pair style meam/alloy/spline requires newton pair on");

  // Need full neighbor list.
  int irequest_full = neighbor->request(this);
  neighbor->requests[irequest_full]->id   = 1;
  neighbor->requests[irequest_full]->half = 0;
  neighbor->requests[irequest_full]->full = 1;
  
}

/* ----------------------------------------------------------------------
   neighbor callback to inform pair style of neighbor list to use (half/full)
------------------------------------------------------------------------- */
void PairGMEAMSpline::init_list( int id, NeighList *ptr )
{

	if (id==1) list = ptr;

}
/* ----------------------------------------------------------------------
   Init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairGMEAMSpline::init_one(int i, int j)
{

  return cutoff_max;
}

/* ----------------------------------------------------------------------
   pack forward comm function for MPI
------------------------------------------------------------------------- */
//int PairGMEAMSpline::pack_forward_comm(int n, int*list, double*buf, int pbc_flag, int*pbc)
//{
////	int* list_iter = list;
////	int* list_iter_end = list + n;
////	while( list_iter != list_iter_end )
////		*buf++ = Uprime_values[*list_iter++];
//	return 0;
//}
//
///* ----------------------------------------------------------------------
//   unpack forward comm function for MPI
//------------------------------------------------------------------------- */
//void PairGMEAMSpline::unpack_forward_comm(int n, int first, double *buf)
//{
////	memcpy(&Uprime_values[first], buf, n * sizeof(buf[0]));
//	return;
//}
//
///* ----------------------------------------------------------------------
//   pack reverse comm function for MPI
//------------------------------------------------------------------------- */
//int PairGMEAMSpline::pack_reverse_comm(int n, int first, double *buf)
//{
//	return 0;
//}
//
///* ----------------------------------------------------------------------
//   unpack reverse comm function for MPI
//------------------------------------------------------------------------- */
//void PairGMEAMSpline::unpack_reverse_comm(int n, int *list, double *buf)
//{
//	return;
//}
//
/* ----------------------------------------------------------------------
   unpack reverse comm function for MPI
------------------------------------------------------------------------- */
//double PairGMEAMSpline::memory_usage()
//{
//	return nmax * sizeof(double);
//}

/* ----------------------------------------------------------------------
   Set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairGMEAMSpline::read_file(std::string filename)
{
  if(comm->me == 0) {
    std::ifstream ifs;
    ifs.open(filename.c_str(), std::ifstream::in);

    if(!ifs) {
      std::stringstream oss;
      oss << "Cannot open spline EAM potential file " << filename;
      std::string error_str = oss.str();
      error->one(FLERR,error_str.c_str());
    }

    // Skip first line of file
    std::string tmp_line;
    std::getline(ifs, tmp_line);

    // Read in potential type
    // WARNING: For now this is useless except for testing scripts
    std::string pot_type;
    std::getline(ifs, pot_type);
    int nfns = 0;

    // Read in potentials
    int ntypes = atom->ntypes;

    // Read in phi: phi_aa, phi_ab, phi_ba, phi_bb
    int nphi = ntypes * ntypes;
    phi.resize(nphi, Spline(lmp));
    for (int i = 0; i < ntypes; ++i) {
      for (int j = i; j < ntypes; ++j) {
        // Try reading in basis function
        try {
          std::string basis_type;
          std::getline(ifs, basis_type);
          ifs >> phi[i*ntypes + j];
          ++nfns; // count number of fns in potential
//	  phi[i*ntypes + j].print_spline(std::cout);
        } catch(ErrorSpline& ex) {
          std::stringstream oss;
          oss << ex.what() << " - phi_pot=" << nfns << " file=" << filename;
          std::string error_str = oss.str();
          error->one(FLERR,error_str.c_str());
        }

        // keep symmetry: phi_ij = phi_ji
        if ( i != j ) phi[j*ntypes + i] = phi[i*ntypes + j];
      }
    }

    // Read in rho: rho_a, rho_b
    int nrho = ntypes;
    rho.resize(nrho, Spline(lmp));
    for (int i = 0; i < nrho; ++i) {
      // Try reading in basis function
      try {
        std::string basis_type;
        std::getline(ifs, basis_type);
        ifs >> rho[i];
        ++nfns; // count number of fns in potential
//	rho[i].print_spline(std::cout);
      } catch(ErrorSpline& ex) {
        std::stringstream oss;
        oss << ex.what() << " - rho_pot=" << nfns << " file=" << filename;
        std::string error_str = oss.str();
        error->one(FLERR,error_str.c_str());
      }
    }

    // Read in U: U_a, U_b
    int nu = ntypes;
    u.resize(nu, Spline(lmp));
    for (int i = 0; i < nu; ++i) {
      // Try reading in basis function
      try {
        std::string basis_type;
        std::getline(ifs, basis_type);
        ifs >> u[i];
        ++nfns; // count number of fns in potential
//	u[i].print_spline(std::cout);
      } catch(ErrorSpline& ex) {
        std::stringstream oss;
        oss << ex.what() << " - u_pot=" << nfns << " file=" << filename;
        std::string error_str = oss.str();
        error->one(FLERR,error_str.c_str());
      }
    }
    
    // Read in f: f_aa, f_ab, f_ba, f_bb
    int nf = ntypes * ntypes;
    f.resize(nf, Spline(lmp));
    for (int i = 0; i < ntypes; ++i) {
      for (int j = i; j < ntypes; ++j) {
        // Try reading in basis function
        try {
          std::string basis_type;
          std::getline(ifs, basis_type);
          ifs >> f[i*ntypes + j];
          ++nfns; // count number of fns in potential
	//  f[i*ntypes + j].print_spline(std::cout);
        } catch(ErrorSpline& ex) {
          std::stringstream oss;
          oss << ex.what() << " - f_pot=" << nfns << " file=" << filename;
          std::string error_str = oss.str();
          error->one(FLERR,error_str.c_str());
        }

        // keep symmetry: f_ij = f_ji
        if ( i != j ) f[j*ntypes + i] = f[i*ntypes + j];
      }
    }

    // Read in g:
    // 2 elements: g_aaa, g_aab, g_abb, g_baa, g_bab, g_bbb
    // 3 elements: g_aaa, g_aac, g_abb, g_abc, g_acc, g_baa, g_bac, g_bbb, g_bbc, g_bcc, g_caa, g_cac, g_cbb, g_cbc, g_ccc
    // NOTE: ijk NOTATION GIVES CENTRAL ATOM FOLLOWED BY PERIPHERAL ATOMS
    int ng = ntypes * ntypes * ntypes;
    g.resize(ng, Spline(lmp));
    for (int i = 0; i < ntypes; ++i) {		// symmetry applies to boundary atoms, not central
      for (int j = 0; j < ntypes; ++j) {		// so loop over all j values
        for (int k = j; k < ntypes; ++k) {		// and one corner of each cross-section matrix of the 3d array
          // Try reading in basis function
          try {
            std::string basis_type;
            std::getline(ifs, basis_type);
            ifs >> g[i*ntypes*ntypes + j*ntypes + k];
            ++nfns; // count number of fns in potential
          } catch(ErrorSpline& ex) {
            std::stringstream oss;
            oss << ex.what() << " - g_pot=" << nfns << " file=" << filename;
            std::string error_str = oss.str();
            error->one(FLERR,error_str.c_str());
          }
          // keep symmetry: u_ijk = u_ikj
          //if ( j != k ) g[i*ntypes*(ntypes+1)/2 + k*ntypes + j - k*(k+1)/2] = g[i*ntypes*(ntypes+1)/2 + j*ntypes + k - j*(j+1)/2];
          if ( j != k ) g[i*ntypes*ntypes + k*ntypes + j] = g[i*ntypes*ntypes + j*ntypes + k];
        }
      }
    }

    ifs.close();

  } // comm->me == 0

  // Communicate potentials
  int nphi = phi.size();
  int nrho = rho.size();
  int nu   = u.size();
  int nf   = f.size();
  int ng   = g.size();
  MPI_Bcast(&nphi, 1, MPI_INT, 0, world);
  MPI_Bcast(&nrho, 1, MPI_INT, 0, world);
  MPI_Bcast(&nu  , 1, MPI_INT, 0, world);
  MPI_Bcast(&nf  , 1, MPI_INT, 0, world);
  MPI_Bcast(&ng  , 1, MPI_INT, 0, world);
  if (comm->me != 0) phi.resize(nphi, Spline(lmp));
  if (comm->me != 0) rho.resize(nrho, Spline(lmp));
  if (comm->me != 0) u.resize(nu, Spline(lmp));
  if (comm->me != 0) f.resize(nf, Spline(lmp));
  if (comm->me != 0) g.resize(ng, Spline(lmp));
  for (int i = 0; i < phi.size(); ++i) phi[i].communicate();
  for (int i = 0; i < rho.size(); ++i) rho[i].communicate();
  for (int i = 0; i < u.size();   ++i) u[i].communicate();
  for (int i = 0; i < f.size();   ++i) f[i].communicate();
  for (int i = 0; i < g.size();   ++i) g[i].communicate();

  // compute 'zero-point energies'
//  double junk;
//  for (int i=0; i < u.size(); ++i) zero_atom_energies[i] = u[i].splint_comb(0.0, junk);
  
  //for (int i = 0; i < phi.size(); ++i) phi[i].print_spline(std::cout);
  //for (int i = 0; i < rho.size(); ++i) rho[i].print_spline(std::cout);
  //for (int i = 0; i < u.size();   ++i) u[i].print_spline(std::cout);
  //for (int i = 0; i < f.size();   ++i) f[i].print_spline(std::cout);
  //for (int i = 0; i < g.size();   ++i) g[i].print_spline(std::cout);

  // Determine maximum cutoff radius of all relevant spline functions
  cutoff_max = 0.0;
  for (int i = 0; i < phi.size(); ++i)
    cutoff_max = std::max(cutoff_max, phi[i].get_cutoff());
  for (int i = 0; i < rho.size(); ++i)
    cutoff_max = std::max(cutoff_max, rho[i].get_cutoff());
  for (int i = 0; i < f.size(); ++i)
    cutoff_max = std::max(cutoff_max, f[i].get_cutoff());

  // Set LAMMPS pair interaction flags
  for(int i = 1; i <= atom->ntypes; ++i) {
    for(int j = 1; j <= atom->ntypes; ++j) {
      setflag[i][j] = 1;          // 0/1 = whether each i,j has been set
      cutsq[i][j] = cutoff_max;   // cutoff sq for each atom pair (neighbor.cpp thinks it is cutoff, NOT cutoff^2)
    }
  }
  

  return;
}

