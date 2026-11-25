# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False
"""Thin Cython wrapper around the native C++ engine (polychrom/_native/engine.hpp).

Use :mod:`polychrom.native` instead of this module directly.
"""

import numpy as np

cimport numpy as cnp
from libc.stdint cimport uint64_t
from libcpp cimport bool as cbool

cnp.import_array()


cdef extern from "_native/engine.hpp" namespace "pcn" nogil:
    cdef cppclass Engine:
        Engine(int N) except +
        void set_threads(int n)
        int get_threads()
        void set_masses(const double* m)
        void set_box(cbool pbc, double lx, double ly, double lz)
        void set_positions(const double* p)
        void get_positions(double* p)
        void set_velocities(const double* p)
        void get_velocities(double* p)
        void get_forces(double* p) except +
        void set_velocities_to_temperature(double T, uint64_t seed)
        int add_bond_set(cbool tabulated, cbool pbc)
        void set_harmonic_bonds(int s, int n, const int* i, const int* j, const double* r0, const double* k) except +
        int add_bond_table(int s, double x0, double x1, int nknots, const double* E, const double* D) except +
        void set_tabulated_bonds(int s, int n, const int* i, const int* j, const int* tab) except +
        void clear_bond_tables(int s) except +
        int add_angle_set(cbool pbc)
        int add_angle_table(int s, int nt, const double* Et, const double* Dt, double c_lo, double c_hi, int nc,
                            const double* Ec, const double* Dc) except +
        void set_angles(int s, int n, const int* i, const int* j, const int* k, const int* tab) except +
        void clear_angle_tables(int s) except +
        int add_pair_force(double cutoff, double skin, int ntypes, const int* type) except +
        void set_pair_table(int s, int ta, int tb, double x0, double x1, int nknots, const double* E, const double* D) except +
        void set_pair_exclusions(int s, int n, const int* i, const int* j) except +
        long get_pair_builds(int s) except +
        long get_pair_list_size(int s) except +
        int get_fast_count()
        void get_fast_particles(int* out)
        void set_pair_skin(int s, double skin) except +
        int add_spherical_confinement(int n, const int* particles, double kb, double aa, double t, double tt,
                                      double sign, double cx, double cy, double cz) except +
        void set_spherical_confinement(int s, double kb, double aa, double t, double tt, double sign,
                                       double cx, double cy, double cz) except +
        int add_tether(int n, const int* particles, const double* x0, double kx, double ky, double kz,
                       cbool pbc) except +
        void set_tether_k(int s, double kx, double ky, double kz) except +
        void set_integrator(int type, double temperature, double friction, double dt, double error_tol) except +
        void set_temperature(double t)
        void set_friction(double g)
        void set_step_size(double dt)
        double get_step_size()
        void set_max_step_size(double m)
        void set_error_tolerance(double e)
        void set_seed(uint64_t s)
        double get_time()
        void set_time(double t)
        long long get_step_count()
        double potential_energy() except +
        double kinetic_energy() except +
        void step(int n) except +
        void compute() except +


INTEGRATORS = {"langevin": 0, "langevinmiddle": 1, "variablelangevin": 2, "brownian": 3}


cdef inline cnp.ndarray _d(a, n=None):
    arr = np.ascontiguousarray(a, dtype=np.float64)
    if n is not None and arr.size != n:
        raise ValueError(f"expected {n} values, got {arr.size}")
    return arr


cdef inline cnp.ndarray _i(a, n=None):
    arr = np.ascontiguousarray(a, dtype=np.intc)
    if n is not None and arr.size != n:
        raise ValueError(f"expected {n} values, got {arr.size}")
    return arr


cdef class NativeEngine:
    cdef Engine* e
    cdef readonly int N

    def __cinit__(self, int N):
        self.e = new Engine(N)
        self.N = N

    def __dealloc__(self):
        del self.e

    # configuration -------------------------------------------------------
    def set_threads(self, int n):
        self.e.set_threads(n)

    def get_threads(self):
        return self.e.get_threads()

    def set_masses(self, m):
        cdef cnp.ndarray a = _d(m, self.N)
        self.e.set_masses(<double*> a.data)

    def set_box(self, pbc, lx=0.0, ly=0.0, lz=0.0):
        self.e.set_box(bool(pbc), lx, ly, lz)

    def set_positions(self, p):
        cdef cnp.ndarray a = _d(p, 3 * self.N)
        self.e.set_positions(<double*> a.data)

    def get_positions(self):
        cdef cnp.ndarray a = np.empty((self.N, 3))
        self.e.get_positions(<double*> a.data)
        return a

    def set_velocities(self, p):
        cdef cnp.ndarray a = _d(p, 3 * self.N)
        self.e.set_velocities(<double*> a.data)

    def get_velocities(self):
        cdef cnp.ndarray a = np.empty((self.N, 3))
        self.e.get_velocities(<double*> a.data)
        return a

    def get_forces(self):
        cdef cnp.ndarray a = np.empty((self.N, 3))
        with nogil:
            self.e.get_forces(<double*> a.data)
        return a

    def set_velocities_to_temperature(self, double T, uint64_t seed):
        self.e.set_velocities_to_temperature(T, seed)

    # bonds ---------------------------------------------------------------
    def add_bond_set(self, tabulated, pbc=False):
        return self.e.add_bond_set(bool(tabulated), bool(pbc))

    def set_harmonic_bonds(self, int s, i, j, r0, k):
        cdef cnp.ndarray ai = _i(i)
        cdef int n = ai.size
        cdef cnp.ndarray aj = _i(j, n), ar = _d(r0, n), ak = _d(k, n)
        self.e.set_harmonic_bonds(s, n, <int*> ai.data, <int*> aj.data, <double*> ar.data, <double*> ak.data)

    def add_bond_table(self, int s, double x0, double x1, E, D):
        cdef cnp.ndarray ae = _d(E)
        cdef cnp.ndarray ad = _d(D, ae.size)
        return self.e.add_bond_table(s, x0, x1, ae.size, <double*> ae.data, <double*> ad.data)

    def set_tabulated_bonds(self, int s, i, j, tab):
        cdef cnp.ndarray ai = _i(i)
        cdef int n = ai.size
        cdef cnp.ndarray aj = _i(j, n), at = _i(tab, n)
        self.e.set_tabulated_bonds(s, n, <int*> ai.data, <int*> aj.data, <int*> at.data)

    def clear_bond_tables(self, int s):
        self.e.clear_bond_tables(s)

    # angles ----------------------------------------------------------------
    def add_angle_set(self, pbc=False):
        return self.e.add_angle_set(bool(pbc))

    def add_angle_table(self, int s, Et, Dt, double c_lo=0.0, double c_hi=0.0, Ec=(), Dc=()):
        """E, dE/dtheta on [0, pi]; optionally E, dE/dcos on [c_lo, c_hi]."""
        cdef cnp.ndarray aet = _d(Et)
        cdef cnp.ndarray adt = _d(Dt, aet.size)
        cdef cnp.ndarray aec = _d(Ec)
        cdef cnp.ndarray adc = _d(Dc, aec.size)
        return self.e.add_angle_table(s, aet.size, <double*> aet.data, <double*> adt.data, c_lo, c_hi, aec.size,
                                      <double*> aec.data, <double*> adc.data)

    def set_angles(self, int s, i, j, k, tab):
        cdef cnp.ndarray ai = _i(i)
        cdef int n = ai.size
        cdef cnp.ndarray aj = _i(j, n), ak = _i(k, n), at = _i(tab, n)
        self.e.set_angles(s, n, <int*> ai.data, <int*> aj.data, <int*> ak.data, <int*> at.data)

    def clear_angle_tables(self, int s):
        self.e.clear_angle_tables(s)

    # pair potentials ---------------------------------------------------------
    def add_pair_force(self, double cutoff, double skin, int ntypes, types):
        cdef cnp.ndarray at = _i(types, self.N)
        return self.e.add_pair_force(cutoff, skin, ntypes, <int*> at.data)

    def set_pair_table(self, int s, int ta, int tb, double x0, double x1, E, D):
        cdef cnp.ndarray ae = _d(E)
        cdef cnp.ndarray ad = _d(D, ae.size)
        self.e.set_pair_table(s, ta, tb, x0, x1, ae.size, <double*> ae.data, <double*> ad.data)

    def set_pair_exclusions(self, int s, i, j):
        cdef cnp.ndarray ai = _i(i)
        cdef int n = ai.size
        cdef cnp.ndarray aj = _i(j, n)
        self.e.set_pair_exclusions(s, n, <int*> ai.data, <int*> aj.data)

    def get_pair_builds(self, int s):
        return self.e.get_pair_builds(s)

    def get_pair_list_size(self, int s):
        return self.e.get_pair_list_size(s)

    def get_fast_count(self):
        """Particles that moved more than half the skin since the last list build."""
        return self.e.get_fast_count()

    def get_fast_particles(self):
        cdef cnp.ndarray a = np.empty(self.e.get_fast_count(), dtype=np.intc)
        if a.size:
            self.e.get_fast_particles(<int*> a.data)
        return a

    def set_pair_skin(self, int s, double skin):
        self.e.set_pair_skin(s, skin)

    # external ----------------------------------------------------------------
    def add_spherical_confinement(self, particles, double kb, double aa, double t, double tt, double sign,
                                  double cx, double cy, double cz):
        cdef cnp.ndarray ap = _i(particles)
        return self.e.add_spherical_confinement(ap.size, <int*> ap.data, kb, aa, t, tt, sign, cx, cy, cz)

    def set_spherical_confinement(self, int s, double kb, double aa, double t, double tt, double sign,
                                  double cx, double cy, double cz):
        self.e.set_spherical_confinement(s, kb, aa, t, tt, sign, cx, cy, cz)

    def add_tether(self, particles, x0, double kx, double ky, double kz, pbc=False):
        cdef cnp.ndarray ap = _i(particles)
        cdef cnp.ndarray ax = _d(x0, 3 * ap.size)
        return self.e.add_tether(ap.size, <int*> ap.data, <double*> ax.data, kx, ky, kz, bool(pbc))

    def set_tether_k(self, int s, double kx, double ky, double kz):
        self.e.set_tether_k(s, kx, ky, kz)

    # integrator ----------------------------------------------------------------
    def set_integrator(self, kind, double temperature, double friction, double dt=0.0, double error_tol=0.0):
        self.e.set_integrator(INTEGRATORS[kind.lower()], temperature, friction, dt, error_tol)

    def set_temperature(self, double t):
        self.e.set_temperature(t)

    def set_friction(self, double g):
        self.e.set_friction(g)

    def set_step_size(self, double dt):
        self.e.set_step_size(dt)

    def get_step_size(self):
        return self.e.get_step_size()

    def set_error_tolerance(self, double e):
        self.e.set_error_tolerance(e)

    def set_max_step_size(self, double m):
        self.e.set_max_step_size(m)

    def set_seed(self, uint64_t s):
        self.e.set_seed(s)

    def get_time(self):
        return self.e.get_time()

    def set_time(self, double t):
        self.e.set_time(t)

    def get_step_count(self):
        return self.e.get_step_count()

    def potential_energy(self):
        cdef double r
        with nogil:
            r = self.e.potential_energy()
        return r

    def kinetic_energy(self):
        cdef double r
        with nogil:
            r = self.e.kinetic_energy()
        return r

    def step(self, int n):
        with nogil:
            self.e.step(n)

    def compute(self):
        with nogil:
            self.e.compute()
