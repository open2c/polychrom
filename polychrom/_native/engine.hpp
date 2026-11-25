// Native CPU engine for polychrom.
//
// A small, specialised molecular dynamics engine for coarse-grained polymer
// models, used by ``Simulation(platform="native")``.  It reproduces the subset
// of OpenMM that polychrom uses (harmonic and tabulated bonds, tabulated
// angles, tabulated short-range pair potentials with exclusions, spherical
// confinement, tethers, and OpenMM's Langevin-type integrators) with data
// structures tuned for polymer systems on multi-core CPUs:
//
//  * one OpenMP parallel region per ``step(n)`` call; the phases of a time step
//    are work-sharing loops separated by barriers, not separate fork/joins;
//  * Verlet neighbour lists with a skin.  Entries are packed into 32 bits
//    (particle index and periodic image code), so the inner loop needs no
//    rounding.  Particles that have moved more than half the skin since the
//    last build (the "fast set", e.g. beads pulled by a freshly placed loop
//    extrusion bond) get their interactions from a direct search every step;
//    lists are rebuilt only when the fast set grows beyond max(64, N/32);
//  * full (i, j and j, i) neighbour lists, so pair forces are accumulated by the
//    owning thread without atomics or per-thread force buffers;
//  * bonded terms evaluated in parallel into a contribution buffer and gathered
//    per particle through a CSR index;
//  * a counter-based Ziggurat Gaussian generator keyed by (seed, step,
//    coordinate).
//
// Forces and trajectories therefore do not depend on the number of threads
// (only the summation order of the reported potential energy does).
//
// Units follow OpenMM: nm, ps, amu, kJ/mol, K.

#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace pcn {

// Same value as OpenMM's BOLTZ (kJ/mol/K).
constexpr double BOLTZ = 0.0083144626181532395;

// Whether this CPU (and OS) supports the instructions of the x86-64-v3 build of
// the engine (AVX2, FMA, BMI2).  Both builds use -ffp-contract=off, so they
// produce the same results.
static inline bool cpu_has_x86_64_v3() {
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("bmi2");
#else
    return false;
#endif
}

static inline int thread_id() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}
static inline int team_size() {
#ifdef _OPENMP
    return omp_get_num_threads();
#else
    return 1;
#endif
}

// Round to nearest without a libm call (valid for |t| < 2^51).
static inline double round_fast(double t) {
    const double M = 6755399441055744.0;  // 2^52 + 2^51
    return (t + M) - M;
}

// ---------------------------------------------------------------------------
// Cubic Hermite table of a function of one variable (value and derivative at
// uniformly spaced knots), stored as per-interval polynomial coefficients in
// t = (x - x_k) / h.  Outside [x0, x1] the function is extrapolated linearly.
// ---------------------------------------------------------------------------
struct Table {
    double x0 = 0, x1 = 0, h = 1, inv_h = 1;
    int n = 0;  // number of intervals
    std::vector<double> c;
    double e_lo = 0, d_lo = 0, e_hi = 0, d_hi = 0;

    void build(double x0_, double x1_, int nknots, const double* E, const double* D) {
        if (nknots < 2 || !(x1_ > x0_)) throw std::invalid_argument("Table: need >= 2 knots and x1 > x0");
        for (int k = 0; k < nknots; ++k)
            if (!std::isfinite(E[k]) || !std::isfinite(D[k])) throw std::invalid_argument("Table: non-finite value");
        x0 = x0_;
        x1 = x1_;
        n = nknots - 1;
        h = (x1 - x0) / n;
        inv_h = 1.0 / h;
        c.assign(4 * (size_t)n, 0.0);
        for (int k = 0; k < n; ++k) {
            const double e0 = E[k], e1 = E[k + 1], m0 = D[k] * h, m1 = D[k + 1] * h;
            c[4 * k + 0] = e0;
            c[4 * k + 1] = m0;
            c[4 * k + 2] = -3 * e0 - 2 * m0 + 3 * e1 - m1;
            c[4 * k + 3] = 2 * e0 + m0 - 2 * e1 + m1;
        }
        e_lo = E[0];
        d_lo = D[0];
        e_hi = E[nknots - 1];
        d_hi = D[nknots - 1];
    }

    // For pair tables on [0, x1]: no range checks, x must be in [0, x1].
    inline void eval_in_range(double x, double& e, double& de) const {
        const double u = (x - x0) * inv_h;
        int k = (int)u;
        k = k < n ? k : n - 1;
        const double t = u - k;
        const double* p = &c[4 * (size_t)k];
        e = p[0] + t * (p[1] + t * (p[2] + t * p[3]));
        de = (p[1] + t * (2.0 * p[2] + t * 3.0 * p[3])) * inv_h;
    }

    inline void eval(double x, double& e, double& de) const {
        const double u = (x - x0) * inv_h;
        if (u < 0.0) {
            de = d_lo;
            e = e_lo + d_lo * (x - x0);
            return;
        }
        const int k = (int)u;
        if (k >= n) {
            de = d_hi;
            e = e_hi + d_hi * (x - x1);
            return;
        }
        const double t = u - k;
        const double* p = &c[4 * (size_t)k];
        e = p[0] + t * (p[1] + t * (p[2] + t * p[3]));
        de = (p[1] + t * (2.0 * p[2] + t * 3.0 * p[3])) * inv_h;
    }
};

// ---------------------------------------------------------------------------
// Random numbers: splitmix64 finaliser as a counter-based hash, and the
// Marsaglia-Tsang (2000) Ziggurat for normal deviates (128 layers).  Layer
// index and value come from disjoint bits of one 64-bit draw.
// ---------------------------------------------------------------------------
static inline uint64_t mix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static inline double u01(uint64_t r) {  // (0, 1]
    return ((r >> 11) + 1) * (1.0 / 9007199254740992.0);
}

struct ZigguratTables {
    uint32_t kn[128];
    double wn[128], fn[128];
    ZigguratTables() {
        const double m1 = 2147483648.0;
        double dn = 3.442619855899, tn = dn;
        const double vn = 9.91256303526217e-3;
        const double q = vn / std::exp(-.5 * dn * dn);
        kn[0] = (uint32_t)((dn / q) * m1);
        kn[1] = 0;
        wn[0] = q / m1;
        wn[127] = dn / m1;
        fn[0] = 1.;
        fn[127] = std::exp(-.5 * dn * dn);
        for (int i = 126; i >= 1; i--) {
            dn = std::sqrt(-2. * std::log(vn / dn + std::exp(-.5 * dn * dn)));
            kn[i + 1] = (uint32_t)((dn / tn) * m1);
            tn = dn;
            fn[i] = std::exp(-.5 * dn * dn);
            wn[i] = dn / m1;
        }
    }
};

static inline const ZigguratTables& zig() {
    static const ZigguratTables t;
    return t;
}

// Standard normal deviate from the stream that starts at state `st`.
static inline double znormal(uint64_t st) {
    const ZigguratTables& Z = zig();
    for (;;) {
        st += 0x9E3779B97F4A7C15ull;
        const uint64_t r = mix64(st);
        const int iz = (int)(r & 127);
        const int32_t hz = (int32_t)(uint32_t)(r >> 32);
        const uint32_t ahz = hz < 0 ? (uint32_t)(-(int64_t)hz) : (uint32_t)hz;
        if (ahz < Z.kn[iz]) return hz * Z.wn[iz];
        if (iz == 0) {  // tail beyond R
            const double R = 3.442619855899;
            double xx, yy;
            do {
                st += 0x9E3779B97F4A7C15ull;
                xx = -std::log(u01(mix64(st))) / R;
                st += 0x9E3779B97F4A7C15ull;
                yy = -std::log(u01(mix64(st)));
            } while (yy + yy < xx * xx);
            return hz > 0 ? R + xx : -R - xx;
        }
        const double xx = hz * Z.wn[iz];
        st += 0x9E3779B97F4A7C15ull;
        if (Z.fn[iz] + u01(mix64(st)) * (Z.fn[iz - 1] - Z.fn[iz]) < std::exp(-.5 * xx * xx)) return xx;
    }
}

// ---------------------------------------------------------------------------
// Force terms
// ---------------------------------------------------------------------------
struct BondSet {  // harmonic (k, r0) or tabulated E(r)
    bool tabulated = false;
    bool pbc = false;
    std::vector<int> i, j;
    std::vector<double> r0, k;  // harmonic
    std::vector<int> tab;       // tabulated: index into tables
    std::vector<Table> tables;
    size_t contrib_offset = 0;
};

struct AngleTable {
    Table t;      // E(theta), theta in [0, pi]
    Table tc;     // E(cos theta), cos theta in [c_lo, c_hi]; used there (avoids atan2)
    double c_lo = 2, c_hi = -2;
};

struct AngleSet {
    bool pbc = false;
    std::vector<int> i, j, k, tab;
    std::vector<AngleTable> tables;
    size_t contrib_offset = 0;
};

constexpr int NB_SHIFT_BITS = 27;
constexpr uint32_t NB_INDEX_MASK = (1u << NB_SHIFT_BITS) - 1;
constexpr int MAX_PARTICLES = 1 << NB_SHIFT_BITS;

struct PairForce {
    double cutoff = 1.0, skin = 0.3;
    int ntypes = 1;
    std::vector<int> type;      // per particle
    std::vector<Table> tables;  // ntypes * ntypes
    std::vector<int> excl_start, excl;  // CSR of sorted exclusions, symmetric
    // Verlet list (full), entries = index | image_code << 27
    std::vector<int> nb_start;
    std::vector<uint32_t> nb;
    bool need_rebuild = true;
    long n_builds = 0;
};

struct SphericalConfinement {  // polychrom.forces.spherical_confinement
    std::vector<int> particles;
    double kb, aa, t, tt, sign, cx, cy, cz;
};

struct Tether {  // polychrom.forces.tether_particles: kx (x-x0)^2 + ky (y-y0)^2 + kz (z-z0)^2
    std::vector<int> particles;
    std::vector<double> x0;  // 3 per particle
    double k[3] = {0, 0, 0};
    bool pbc = false;
};

// LANGEVIN is LANGEVIN_MIDDLE: in OpenMM 8, LangevinIntegrator subclasses LangevinMiddleIntegrator.
enum IntegratorType { LANGEVIN = 0, LANGEVIN_MIDDLE = 1, VARIABLE_LANGEVIN = 2, BROWNIAN = 3 };

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------
class Engine {
   public:
    explicit Engine(int N_) : N(N_) {
        if (N <= 0) throw std::invalid_argument("Engine: N must be positive");
        if (N >= MAX_PARTICLES) throw std::invalid_argument("Engine: too many particles");
        x.assign(3 * (size_t)N, 0.0);
        v.assign(3 * (size_t)N, 0.0);
        f.assign(3 * (size_t)N, 0.0);
        x_ref.assign(3 * (size_t)N, 0.0);
        xw.assign(3 * (size_t)N, 0.0);
        img.assign(3 * (size_t)N, 0.0);
        mass.assign(N, 100.0);
        inv_mass.assign(N, 0.01);
        (void)zig();
    }

    // ---- configuration ---------------------------------------------------
    void set_threads(int n) {
#ifdef _OPENMP
        if (n > 0) omp_set_num_threads(n);
#endif
        (void)n;
    }
    int get_threads() const {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }

    void set_masses(const double* m) {
        for (int p = 0; p < N; ++p) {
            mass[p] = m[p];
            inv_mass[p] = m[p] > 0 ? 1.0 / m[p] : 0.0;
        }
        invalidate();
    }

    void set_box(bool use_pbc, double lx, double ly, double lz) {
        if (use_pbc && !(lx > 0 && ly > 0 && lz > 0)) throw std::invalid_argument("box sizes must be positive");
        pbc = use_pbc;
        box[0] = lx;
        box[1] = ly;
        box[2] = lz;
        invalidate();
    }

    void set_positions(const double* p) {
        for (size_t c = 0; c < 3 * (size_t)N; ++c)
            if (!std::isfinite(p[c])) throw std::invalid_argument("non-finite position");
        std::copy(p, p + 3 * (size_t)N, x.begin());
        invalidate();
    }
    void get_positions(double* p) const { std::copy(x.begin(), x.end(), p); }
    void set_velocities(const double* p) { std::copy(p, p + 3 * (size_t)N, v.begin()); }
    void get_velocities(double* p) const { std::copy(v.begin(), v.end(), p); }
    void get_forces(double* p) {
        compute();
        std::copy(f.begin(), f.end(), p);
    }

    void set_velocities_to_temperature(double T_, uint64_t seed_) {
        const double kT = BOLTZ * T_;
        const uint64_t base = mix64(seed_ ^ 0xD1B54A32D192ED03ull);
        for (int p = 0; p < N; ++p) {
            const double s = std::sqrt(kT * inv_mass[p]);
            for (int d = 0; d < 3; ++d) v[3 * p + d] = s * znormal(mix64(base ^ (uint64_t)(3 * p + d)));
        }
    }

    // Bonds -----------------------------------------------------------------
    int add_bond_set(bool tabulated, bool use_pbc) {
        BondSet b;
        b.tabulated = tabulated;
        b.pbc = use_pbc;
        bonds.push_back(std::move(b));
        topology_dirty = true;
        invalidate();
        return (int)bonds.size() - 1;
    }
    void set_harmonic_bonds(int s, int n, const int* i, const int* j, const double* r0, const double* k) {
        BondSet& b = bonds.at(s);
        if (b.tabulated) throw std::invalid_argument("bond set is tabulated");
        if (n != (int)b.i.size() || !std::equal(i, i + n, b.i.begin()) || !std::equal(j, j + n, b.j.begin())) {
            check_indices(n, i);
            check_indices(n, j);
            b.i.assign(i, i + n);
            b.j.assign(j, j + n);
            topology_dirty = true;
        }
        b.r0.assign(r0, r0 + n);
        b.k.assign(k, k + n);
        invalidate();
    }
    int add_bond_table(int s, double x0, double x1, int nknots, const double* E, const double* D) {
        BondSet& b = bonds.at(s);
        b.tables.emplace_back();
        b.tables.back().build(x0, x1, nknots, E, D);
        invalidate();
        return (int)b.tables.size() - 1;
    }
    void set_tabulated_bonds(int s, int n, const int* i, const int* j, const int* tab) {
        BondSet& b = bonds.at(s);
        if (!b.tabulated) throw std::invalid_argument("bond set is harmonic");
        check_indices(n, i);
        check_indices(n, j);
        for (int q = 0; q < n; ++q)
            if (tab[q] < 0 || tab[q] >= (int)b.tables.size()) throw std::out_of_range("bond table index");
        b.i.assign(i, i + n);
        b.j.assign(j, j + n);
        b.tab.assign(tab, tab + n);
        topology_dirty = true;
        invalidate();
    }
    void clear_bond_tables(int s) {
        bonds.at(s).tables.clear();
        invalidate();
    }

    // Angles ----------------------------------------------------------------
    int add_angle_set(bool use_pbc) {
        AngleSet a;
        a.pbc = use_pbc;
        angles.push_back(std::move(a));
        topology_dirty = true;
        invalidate();
        return (int)angles.size() - 1;
    }
    // Et, Dt: E and dE/dtheta on uniform knots in [0, pi].  Ec, Dc: E and dE/dcos
    // on uniform knots in [c_lo, c_hi] (empty if nc == 0), used for cos(theta) in
    // that range.
    int add_angle_table(int s, int nt, const double* Et, const double* Dt, double c_lo, double c_hi, int nc,
                        const double* Ec, const double* Dc) {
        AngleSet& a = angles.at(s);
        AngleTable at;
        at.t.build(0.0, M_PI, nt, Et, Dt);
        if (nc > 0) {
            if (!(c_lo >= -1.0 && c_hi <= 1.0 && c_hi > c_lo)) throw std::invalid_argument("bad cos range");
            at.tc.build(c_lo, c_hi, nc, Ec, Dc);
            at.c_lo = c_lo;
            at.c_hi = c_hi;
        }
        a.tables.push_back(std::move(at));
        invalidate();
        return (int)a.tables.size() - 1;
    }
    void set_angles(int s, int n, const int* i, const int* j, const int* k, const int* tab) {
        AngleSet& a = angles.at(s);
        check_indices(n, i);
        check_indices(n, j);
        check_indices(n, k);
        for (int q = 0; q < n; ++q)
            if (tab[q] < 0 || tab[q] >= (int)a.tables.size()) throw std::out_of_range("angle table index");
        a.i.assign(i, i + n);
        a.j.assign(j, j + n);
        a.k.assign(k, k + n);
        a.tab.assign(tab, tab + n);
        topology_dirty = true;
        invalidate();
    }
    void clear_angle_tables(int s) {
        angles.at(s).tables.clear();
        invalidate();
    }

    // Pair potentials ---------------------------------------------------------
    int add_pair_force(double cutoff, double skin, int ntypes, const int* type) {
        if (!(cutoff > 0)) throw std::invalid_argument("cutoff must be positive");
        if (ntypes < 1) throw std::invalid_argument("ntypes must be >= 1");
        PairForce pf;
        pf.cutoff = cutoff;
        pf.skin = std::max(0.0, skin);
        pf.ntypes = ntypes;
        pf.type.assign(type, type + N);
        for (int t : pf.type)
            if (t < 0 || t >= ntypes) throw std::out_of_range("particle type");
        pf.tables.resize((size_t)ntypes * ntypes);
        pf.excl_start.assign(N + 1, 0);
        pairs.push_back(std::move(pf));
        invalidate();
        return (int)pairs.size() - 1;
    }
    void set_pair_table(int s, int ta, int tb, double x0, double x1, int nknots, const double* E, const double* D) {
        PairForce& pf = pairs.at(s);
        if (ta < 0 || tb < 0 || ta >= pf.ntypes || tb >= pf.ntypes) throw std::out_of_range("pair type");
        pf.tables[(size_t)ta * pf.ntypes + tb].build(x0, x1, nknots, E, D);
        invalidate();
    }
    void set_pair_skin(int s, double skin) {
        pairs.at(s).skin = std::max(0.0, skin);
        invalidate();
    }
    void set_pair_exclusions(int s, int n, const int* i, const int* j) {
        PairForce& pf = pairs.at(s);
        check_indices(n, i);
        check_indices(n, j);
        std::vector<int> cnt(N + 1, 0);
        for (int q = 0; q < n; ++q) {
            if (i[q] == j[q]) continue;
            cnt[i[q] + 1]++;
            cnt[j[q] + 1]++;
        }
        for (int p = 0; p < N; ++p) cnt[p + 1] += cnt[p];
        pf.excl_start = cnt;
        pf.excl.assign(cnt[N], 0);
        std::vector<int> fill(cnt.begin(), cnt.end() - 1);
        for (int q = 0; q < n; ++q) {
            if (i[q] == j[q]) continue;
            pf.excl[fill[i[q]]++] = j[q];
            pf.excl[fill[j[q]]++] = i[q];
        }
        for (int p = 0; p < N; ++p) {
            auto b = pf.excl.begin() + pf.excl_start[p], e = pf.excl.begin() + pf.excl_start[p + 1];
            std::sort(b, e);
        }
        invalidate();
    }
    long get_pair_builds(int s) const { return pairs.at(s).n_builds; }
    long get_pair_list_size(int s) const { return (long)pairs.at(s).nb.size(); }
    int get_fast_count() const { return nfast; }
    void get_fast_particles(int* out) const { std::copy(fast_list.begin(), fast_list.end(), out); }

    // External forces ---------------------------------------------------------
    int add_spherical_confinement(int n, const int* particles, double kb, double aa, double t, double tt,
                                  double sign, double cx, double cy, double cz) {
        check_indices(n, particles);
        SphericalConfinement sc{std::vector<int>(particles, particles + n), kb, aa, t, tt, sign, cx, cy, cz};
        spheres.push_back(std::move(sc));
        invalidate();
        return (int)spheres.size() - 1;
    }
    void set_spherical_confinement(int s, double kb, double aa, double t, double tt, double sign, double cx,
                                   double cy, double cz) {
        SphericalConfinement& sc = spheres.at(s);
        sc.kb = kb;
        sc.aa = aa;
        sc.t = t;
        sc.tt = tt;
        sc.sign = sign;
        sc.cx = cx;
        sc.cy = cy;
        sc.cz = cz;
        invalidate();
    }
    int add_tether(int n, const int* particles, const double* x0, double kx, double ky, double kz, bool use_pbc) {
        check_indices(n, particles);
        Tether t;
        t.particles.assign(particles, particles + n);
        t.x0.assign(x0, x0 + 3 * (size_t)n);
        t.k[0] = kx;
        t.k[1] = ky;
        t.k[2] = kz;
        t.pbc = use_pbc;
        tethers.push_back(std::move(t));
        invalidate();
        return (int)tethers.size() - 1;
    }
    void set_tether_k(int s, double kx, double ky, double kz) {
        Tether& t = tethers.at(s);
        t.k[0] = kx;
        t.k[1] = ky;
        t.k[2] = kz;
        invalidate();
    }

    // Integrator ----------------------------------------------------------------
    void set_integrator(int type, double temperature, double friction, double dt, double error_tol) {
        if (type < LANGEVIN || type > BROWNIAN) throw std::invalid_argument("unknown integrator type");
        if (type == BROWNIAN && !(friction > 0)) throw std::invalid_argument("Brownian dynamics needs friction > 0");
        integrator = (IntegratorType)type;
        T = temperature;
        gamma = friction;
        error_tol_ = error_tol;
        if (integrator == VARIABLE_LANGEVIN) {
            if (!(error_tol > 0)) throw std::invalid_argument("error tolerance must be positive");
            dt_ = 0.0;
        } else {
            if (!(dt > 0)) throw std::invalid_argument("step size must be positive");
            dt_ = dt;
        }
    }
    void set_temperature(double t) { T = t; }
    void set_friction(double g) { gamma = g; }
    void set_step_size(double dt) { dt_ = dt; }
    double get_step_size() const { return dt_; }
    void set_error_tolerance(double e) { error_tol_ = e; }
    void set_max_step_size(double m) { max_dt = m > 0 ? m : std::numeric_limits<double>::infinity(); }
    void set_seed(uint64_t s) { seed = s; }
    double get_time() const { return time_; }
    void set_time(double t) { time_ = t; }
    long long get_step_count() const { return nsteps; }

    double potential_energy() {
        compute();
        return epot;
    }

    // Kinetic energy as reported by OpenMM's Reference platform
    // (computeShiftedKineticEnergy): VariableLangevin shifts the velocities by
    // half a step, LangevinMiddle (= Langevin) and Brownian do not.
    double kinetic_energy() {
        double shift = 0.0;
        if (integrator == VARIABLE_LANGEVIN) shift = 0.5 * dt_;
        if (shift != 0.0) compute();
        double ek = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : ek)
        for (int p = 0; p < N; ++p) {
            double s = 0;
            for (int d = 0; d < 3; ++d) {
                const double vv = v[3 * p + d] + shift * f[3 * p + d] * inv_mass[p];
                s += vv * vv;
            }
            ek += 0.5 * mass[p] * s;
        }
        return ek;
    }

    void compute() {
        if (forces_valid) return;
        prepare();
        stepping = false;
#pragma omp parallel
        forces_inner();
        check_error();
    }

    void step(int n) {
        if (n <= 0) return;
        prepare();
        stepping = true;
        dt_selected = false;
#pragma omp parallel
        {
            for (int s = 0; s < n; ++s) {
                forces_inner();
                integrate_inner();
                if (error_flag.load(std::memory_order_relaxed)) break;  // same value on all threads after barrier
            }
        }
        stepping = false;
        check_error();
    }

   private:
    int N;
    bool pbc = false;
    double box[3] = {0, 0, 0};
    std::vector<double> x, v, f, mass, inv_mass;
    std::vector<double> x_ref;  // positions at the last neighbour-list build
    std::vector<double> img;    // periodic image offset at the last build (multiple of box)
    std::vector<double> xw;     // x - img, refreshed every force evaluation

    std::vector<BondSet> bonds;
    std::vector<AngleSet> angles;
    std::vector<PairForce> pairs;
    std::vector<SphericalConfinement> spheres;
    std::vector<Tether> tethers;

    bool topology_dirty = true;
    std::vector<double> contrib;
    std::vector<int> gather_start, gather_idx;

    bool forces_valid = false;
    double epot = 0.0;
    bool rebuild_all = true;
    // Fast set: particles that moved more than half the skin since the last build.
    std::vector<uint8_t> fast;  // per particle
    std::vector<int> fast_list;
    std::vector<std::pair<int, int>> fast_binned;  // (current cell, particle), sorted
    int nfast = 0;
    double fast_lim2 = 0.0;  // (skin / 2)^2
    int fast_max = 64;
    struct Contribution {
        int q;
        double fx, fy, fz;
    };
    struct FastSpan {  // contributions of fast_list[a]: thr_scratch[tid].contrib[start, start + count)
        int tid;
        size_t start, count;
    };
    std::vector<FastSpan> fast_span;

    IntegratorType integrator = LANGEVIN_MIDDLE;
    double T = 300.0, gamma = 1.0, dt_ = 0.001, error_tol_ = 0.01;
    double max_dt = std::numeric_limits<double>::infinity();
    double vl_err = 0.0;       // variable Langevin: sum of (f/m)^2 over all coordinates
    bool stepping = false;     // inside step(): select the step size together with the forces
    bool dt_selected = false;  // step size of the current step already selected
    double time_ = 0.0;
    long long nsteps = 0;
    uint64_t seed = 1;

    // per-thread scratch (padded against false sharing)
    static constexpr int PAD = 8;
    std::vector<double> thr;
    // per-block partial sums over particles (independent of the number of threads)
    static constexpr int SUM_BLOCK = 64;
    std::vector<double> block_sum;
    struct ThreadScratch {
        std::vector<double> bx, by, bz;  // candidate buffer of the current cell (capacity, not length)
        std::vector<uint32_t> be;
        std::vector<uint32_t> ent;  // list entries found by this thread (capacity, not length)
        std::vector<std::pair<int, size_t>> seg;  // (particle, offset in ent)
        std::vector<int> fast;                    // fast particles found by this thread
        std::vector<Contribution> contrib;        // forces on other particles from the fast set
    };
    std::vector<ThreadScratch> thr_scratch;

    // neighbour-list scratch
    std::vector<int> cell_of, cell_start, cell_list, nb_count, fill, occupied;
    std::vector<double> sorted_xyz;  // wrapped coordinates in cell_list order: x[N], y[N], z[N]
    double shifts[27][3];
    double grid_lo[3], grid_edge[3];
    int grid_nc[3] = {1, 1, 1};

    std::atomic<int> error_flag{0};
    std::string error_msg;

    void check_indices(int n, const int* idx) const {
        for (int q = 0; q < n; ++q)
            if (idx[q] < 0 || idx[q] >= N) throw std::out_of_range("particle index out of range");
    }

    void invalidate() {
        forces_valid = false;
        for (auto& pf : pairs) pf.need_rebuild = true;
    }

    void set_error(const char* msg) {
        int expected = 0;
        if (error_flag.compare_exchange_strong(expected, 1)) error_msg = msg;
    }
    void check_error() {
        if (error_flag.load()) {
            error_flag = 0;
            forces_valid = false;
            throw std::runtime_error(error_msg);
        }
    }

    inline void min_image(double& dx, double& dy, double& dz) const {
        dx -= box[0] * round_fast(dx / box[0]);
        dy -= box[1] * round_fast(dy / box[1]);
        dz -= box[2] * round_fast(dz / box[2]);
    }

    // Serial preparation before entering a parallel region.
    void prepare() {
        if (topology_dirty) rebuild_topology();
        int nt = get_threads();
        thr.assign((size_t)PAD * (nt + 1), 0.0);
        thr_scratch.resize(nt);
        block_sum.assign((N + SUM_BLOCK - 1) / SUM_BLOCK, 0.0);
        if (pbc)
            for (auto& pf : pairs) {
                const double lmin = std::min(box[0], std::min(box[1], box[2]));
                if (!(pf.cutoff < 0.5 * lmin))
                    throw std::invalid_argument("cutoff must be smaller than half the periodic box");
                if (pf.cutoff + pf.skin >= 0.5 * lmin) pf.skin = std::max(0.0, 0.499 * lmin - pf.cutoff);
            }
        if ((int)fast.size() != N) {
            fast.assign(N, 0);
            fast_list.clear();
            nfast = 0;
            for (auto& pf : pairs) pf.need_rebuild = true;
        }
        double smin = std::numeric_limits<double>::infinity();
        for (auto& pf : pairs) smin = std::min(smin, pf.skin);
        fast_lim2 = pairs.empty() ? std::numeric_limits<double>::infinity() : 0.25 * smin * smin;
        fast_max = std::max(64, N / 32);
        for (int a = 0; a < 27; ++a) {
            const int sx = a / 9 - 1, sy = (a / 3) % 3 - 1, sz = a % 3 - 1;
            shifts[a][0] = sx * box[0];
            shifts[a][1] = sy * box[1];
            shifts[a][2] = sz * box[2];
        }
    }

    // Sum the per-thread value slot into a single number (call from a single thread after a barrier).
    double thr_sum() const {
        double s = 0;
        for (size_t t = 0; t + PAD <= thr.size(); t += PAD) s += thr[t];
        return s;
    }

    // ---- bonded bookkeeping ----------------------------------------------------
    void rebuild_topology() {
        size_t off = 0;
        for (auto& b : bonds) {
            b.contrib_offset = off;
            off += 2 * b.i.size();
        }
        for (auto& a : angles) {
            a.contrib_offset = off;
            off += 3 * a.i.size();
        }
        contrib.assign(3 * off, 0.0);
        std::vector<int> owner(off);
        for (auto& b : bonds)
            for (size_t q = 0; q < b.i.size(); ++q) {
                owner[b.contrib_offset + 2 * q] = b.i[q];
                owner[b.contrib_offset + 2 * q + 1] = b.j[q];
            }
        for (auto& a : angles)
            for (size_t q = 0; q < a.i.size(); ++q) {
                owner[a.contrib_offset + 3 * q] = a.i[q];
                owner[a.contrib_offset + 3 * q + 1] = a.j[q];
                owner[a.contrib_offset + 3 * q + 2] = a.k[q];
            }
        gather_start.assign(N + 1, 0);
        for (size_t c = 0; c < off; ++c) gather_start[owner[c] + 1]++;
        for (int p = 0; p < N; ++p) gather_start[p + 1] += gather_start[p];
        gather_idx.assign(off, 0);
        std::vector<int> fl(gather_start.begin(), gather_start.end() - 1);
        for (size_t c = 0; c < off; ++c) gather_idx[fl[owner[c]]++] = (int)c;
        topology_dirty = false;
    }

    // ---- neighbour lists (called by all threads of a team) -------------------------
    void build_list_inner(PairForce& pf) {
        const int tid = thread_id();
        const double rl = pf.cutoff + pf.skin, rl2 = rl * rl;
        // 1. image offsets and wrapped coordinates
        if (pbc) {
#pragma omp for schedule(static)
            for (int p = 0; p < N; ++p)
                for (int d = 0; d < 3; ++d) {
                    double fl = std::floor(x[3 * p + d] / box[d]);
                    double w = x[3 * p + d] - fl * box[d];
                    if (w >= box[d]) {  // rounding at the upper edge
                        w -= box[d];
                        fl += 1.0;
                    }
                    if (w < 0) w = 0;
                    img[3 * p + d] = fl * box[d];
                    xw[3 * p + d] = w;
                }
        } else {
#pragma omp for schedule(static)
            for (size_t c = 0; c < 3 * (size_t)N; ++c) {
                img[c] = 0.0;
                xw[c] = x[c];
            }
        }
        // 2. grid geometry and counting sort into cells
#pragma omp single
        {
            double len[3];
            if (pbc) {
                for (int d = 0; d < 3; ++d) {
                    grid_lo[d] = 0.0;
                    len[d] = box[d];
                }
            } else {
                for (int d = 0; d < 3; ++d) {
                    double mn = std::numeric_limits<double>::infinity(), mx = -mn;
                    for (int p = 0; p < N; ++p) {
                        mn = std::min(mn, x[3 * p + d]);
                        mx = std::max(mx, x[3 * p + d]);
                    }
                    if (!std::isfinite(mn) || !std::isfinite(mx)) {
                        set_error("non-finite particle coordinates (simulation exploded?)");
                        mn = 0;
                        mx = 1;
                    }
                    grid_lo[d] = mn;
                    len[d] = std::max(mx - mn, rl) * (1 + 1e-12);
                }
            }
            double e = rl;
            const double vol = len[0] * len[1] * len[2];
            e = std::max(e, std::cbrt(vol / (8.0 * N + 27)));  // at most ~8N cells
            long ncell = 1;
            for (int d = 0; d < 3; ++d) {
                grid_nc[d] = std::max(1, (int)std::floor(len[d] / e));
                grid_edge[d] = len[d] / grid_nc[d];
                ncell *= grid_nc[d];
            }
            cell_of.resize(N);
            cell_list.resize(N);
            nb_count.resize(N);
            cell_start.assign(ncell + 1, 0);
        }
        // cell of each particle (parallel), then counting sort (serial)
#pragma omp for schedule(static)
        for (int p = 0; p < N; ++p) {
            int ci[3];
            for (int d = 0; d < 3; ++d) {
                double cd = (xw[3 * p + d] - grid_lo[d]) / grid_edge[d];
                cd = cd > 0 ? cd : 0;  // also maps NaN to 0
                ci[d] = cd < grid_nc[d] - 1 ? (int)cd : grid_nc[d] - 1;
            }
            cell_of[p] = (ci[0] * grid_nc[1] + ci[1]) * grid_nc[2] + ci[2];
        }
#pragma omp single
        {
            const long ncell = (long)cell_start.size() - 1;
            for (int p = 0; p < N; ++p) cell_start[cell_of[p] + 1]++;
            occupied.clear();
            for (long c = 0; c < ncell; ++c) {
                if (cell_start[c + 1]) occupied.push_back((int)c);
                cell_start[c + 1] += cell_start[c];
            }
            fill.assign(cell_start.begin(), cell_start.end() - 1);
            for (int p = 0; p < N; ++p) cell_list[fill[cell_of[p]]++] = p;
            sorted_xyz.resize(3 * (size_t)N);
        }
        // wrapped coordinates in cell order (structure of arrays)
        double* const SX = sorted_xyz.data();
        double* const SY = SX + N;
        double* const SZ = SY + N;
#pragma omp for schedule(static)
        for (int s = 0; s < N; ++s) {
            const int q = cell_list[s];
            SX[s] = xw[3 * q];
            SY[s] = xw[3 * q + 1];
            SZ[s] = xw[3 * q + 2];
        }
        // 3. candidate scan per occupied cell.  The 3x3 columns of the stencil
        // are contiguous runs of cells along z, hence contiguous ranges of the
        // cell-ordered coordinates.  They are copied (shifted to the right
        // periodic image) into one flat buffer that all particles of the cell
        // then filter without branches.
        ThreadScratch& ts = thr_scratch[tid];
        size_t nent = 0;
        ts.seg.clear();
        const int nc0 = grid_nc[0], nc1 = grid_nc[1], nc2 = grid_nc[2];
        const int nocc = (int)occupied.size();
#pragma omp for schedule(dynamic, 32) nowait
        for (int oc = 0; oc < nocc; ++oc) {
            const int c = occupied[oc];
            const int cx = c / (nc1 * nc2), cy = (c / nc2) % nc1, cz = c % nc2;
            int run_lo[27], run_hi[27], run_code[27], nrun = 0, ncand = 0;
            auto add_run = [&](int first_cell, int last_cell, int code) {  // inclusive cell range
                const int lo = cell_start[first_cell], hi = cell_start[last_cell + 1];
                if (lo == hi) return;
                run_lo[nrun] = lo;
                run_hi[nrun] = hi;
                run_code[nrun++] = code;
                ncand += hi - lo;
            };
            for (int ox = -1; ox <= 1; ++ox) {
                int a = cx + ox, sx = 0;
                if (a < 0 || a >= nc0) {
                    if (!pbc) continue;
                    if (a < 0) { a += nc0; sx = -1; } else { a -= nc0; sx = 1; }
                }
                for (int oy = -1; oy <= 1; ++oy) {
                    int b = cy + oy, sy = 0;
                    if (b < 0 || b >= nc1) {
                        if (!pbc) continue;
                        if (b < 0) { b += nc1; sy = -1; } else { b -= nc1; sy = 1; }
                    }
                    const int col = (a * nc1 + b) * nc2;
                    const int code0 = (sx + 1) * 9 + (sy + 1) * 3;  // + (sz + 1)
                    if (cz >= 1 && cz + 1 < nc2) {
                        add_run(col + cz - 1, col + cz + 1, code0 + 1);
                    } else {
                        for (int oz = -1; oz <= 1; ++oz) {
                            int d = cz + oz, sz = 0;
                            if (d < 0 || d >= nc2) {
                                if (!pbc) continue;
                                if (d < 0) { d += nc2; sz = -1; } else { d -= nc2; sz = 1; }
                            }
                            add_run(col + d, col + d, code0 + sz + 1);
                        }
                    }
                }
            }
            // One buffer per cell, so that each particle runs a single filter
            // loop (separate short loops per run cost a branch miss each).
            if ((int)ts.bx.size() < ncand) {
                ts.bx.resize(2 * ncand);
                ts.by.resize(2 * ncand);
                ts.bz.resize(2 * ncand);
                ts.be.resize(2 * ncand);
            }
            double* const BX = ts.bx.data();
            double* const BY = ts.by.data();
            double* const BZ = ts.bz.data();
            uint32_t* const BE = ts.be.data();
            const int* const CL = cell_list.data();
            int nbuf = 0;
            for (int r = 0; r < nrun; ++r) {
                const double* sh = shifts[run_code[r]];
                const double shx = sh[0], shy = sh[1], shz = sh[2];
                const uint32_t code = (uint32_t)run_code[r] << NB_SHIFT_BITS;
                const int lo = run_lo[r], len = run_hi[r] - lo;
                for (int k = 0; k < len; ++k) {
                    BX[nbuf + k] = SX[lo + k] + shx;
                    BY[nbuf + k] = SY[lo + k] + shy;
                    BZ[nbuf + k] = SZ[lo + k] + shz;
                    BE[nbuf + k] = (uint32_t)CL[lo + k] | code;
                }
                nbuf += len;
            }
            for (int si = cell_start[c]; si < cell_start[c + 1]; ++si) {
                const int p = CL[si];
                const double px = SX[si], py = SY[si], pz = SZ[si];
                if (ts.ent.size() < nent + nbuf) ts.ent.resize(2 * (nent + nbuf));
                uint32_t* const out = ts.ent.data() + nent;
                int n = 0;
                for (int k = 0; k < nbuf; ++k) {
                    const double dx = px - BX[k], dy = py - BY[k], dz = pz - BZ[k];
                    out[n] = BE[k];
                    n += (dx * dx + dy * dy + dz * dz < rl2);
                }
                // drop p itself and its exclusions (compared branch-free if few)
                const int* ex_b = pf.excl.data() + pf.excl_start[p];
                const int nex = pf.excl_start[p + 1] - pf.excl_start[p];
                const uint32_t self = (uint32_t)p | (13u << NB_SHIFT_BITS);
                int m = 0;
                if (nex <= 8) {
                    for (int k = 0; k < n; ++k) {
                        const uint32_t ent = out[k];
                        const int q = (int)(ent & NB_INDEX_MASK);
                        bool keep = ent != self;
                        for (int z = 0; z < nex; ++z) keep &= ex_b[z] != q;
                        out[m] = ent;
                        m += keep;
                    }
                } else {
                    for (int k = 0; k < n; ++k) {
                        const uint32_t ent = out[k];
                        if (ent == self || std::binary_search(ex_b, ex_b + nex, (int)(ent & NB_INDEX_MASK))) continue;
                        out[m++] = ent;
                    }
                }
                nb_count[p] = m;
                ts.seg.push_back({p, nent});
                nent += m;
            }
        }
#pragma omp barrier
        // 4. CSR offsets, then each thread scatters its segments
#pragma omp single
        {
            pf.nb_start.resize(N + 1);
            pf.nb_start[0] = 0;
            for (int p = 0; p < N; ++p) pf.nb_start[p + 1] = pf.nb_start[p] + nb_count[p];
            pf.nb.resize(pf.nb_start[N]);
        }
        for (const auto& sg : ts.seg)
            std::copy(ts.ent.begin() + sg.second, ts.ent.begin() + sg.second + nb_count[sg.first],
                      pf.nb.begin() + pf.nb_start[sg.first]);
#pragma omp barrier
        // 5. reference positions
#pragma omp for schedule(static)
        for (size_t c = 0; c < 3 * (size_t)N; ++c) x_ref[c] = x[c];
#pragma omp single
        {
            pf.need_rebuild = false;
            pf.n_builds++;
        }
    }

    // Pair interactions of the fast set (called by all threads of a team, after
    // the list-based pair loop).  Fast-slow pairs are found through the cell
    // grid of the last build: a slow particle moved less than skin/2, so it is
    // within the 27-cell stencil around the fast particle's current cell.
    // Fast-fast pairs are found by binning the fast set into the same grid at
    // its current positions.  Each pair is computed once; forces on the partner
    // are buffered and applied by one thread, in the order of the fast list, so
    // that the result does not depend on which thread handled which particle.
    void fast_set_forces(double& e) {
        const int tid = thread_id();
        const PairForce& pf = pairs[0];
        const double rc2 = pf.cutoff * pf.cutoff;
        const int nc0 = grid_nc[0], nc1 = grid_nc[1], nc2 = grid_nc[2];
        std::vector<Contribution>& out = thr_scratch[tid].contrib;
        out.clear();
        auto current_cell = [&](int p, int* ci) {
            for (int d = 0; d < 3; ++d) {
                double xx = x[3 * p + d];
                if (pbc) xx -= std::floor(xx / box[d]) * box[d];
                double cd = (xx - grid_lo[d]) / grid_edge[d];
                cd = cd > 0 ? cd : 0;
                ci[d] = cd < grid_nc[d] - 1 ? (int)cd : grid_nc[d] - 1;
            }
        };
#pragma omp single
        {
            fast_span.resize(nfast);
            fast_binned.resize(nfast);
            for (int a = 0; a < nfast; ++a) {
                int ci[3];
                current_cell(fast_list[a], ci);
                fast_binned[a] = {(ci[0] * nc1 + ci[1]) * nc2 + ci[2], fast_list[a]};
            }
            std::sort(fast_binned.begin(), fast_binned.end());
        }
        auto pair = [&](int p, int q, double& fx, double& fy, double& fz) {
            double dx = x[3 * p] - x[3 * q], dy = x[3 * p + 1] - x[3 * q + 1], dz = x[3 * p + 2] - x[3 * q + 2];
            if (pbc) min_image(dx, dy, dz);
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= rc2) return;
            const int* ex_b = pf.excl.data() + pf.excl_start[p];
            const int* ex_e = pf.excl.data() + pf.excl_start[p + 1];
            if (ex_b != ex_e && std::binary_search(ex_b, ex_e, q)) return;
            const double r = std::sqrt(r2);
            double en, de;
            pf.tables[(size_t)pf.type[p] * pf.ntypes + pf.type[q]].eval_in_range(r, en, de);
            e += en;
            const double sc = r > 0 ? -de / r : 0.0;
            fx += sc * dx;
            fy += sc * dy;
            fz += sc * dz;
            out.push_back({q, -sc * dx, -sc * dy, -sc * dz});
        };
#pragma omp for schedule(dynamic, 4)
        for (int a = 0; a < nfast; ++a) {
            const int p = fast_list[a];
            const size_t out0 = out.size();
            double fx = 0, fy = 0, fz = 0;
            int ci[3];
            current_cell(p, ci);
            for (int ox = -1; ox <= 1; ++ox) {
                int ca = ci[0] + ox;
                if (ca < 0 || ca >= nc0) {
                    if (!pbc) continue;
                    ca = (ca + nc0) % nc0;
                }
                for (int oy = -1; oy <= 1; ++oy) {
                    int cb = ci[1] + oy;
                    if (cb < 0 || cb >= nc1) {
                        if (!pbc) continue;
                        cb = (cb + nc1) % nc1;
                    }
                    for (int oz = -1; oz <= 1; ++oz) {
                        int cc = ci[2] + oz;
                        if (cc < 0 || cc >= nc2) {
                            if (!pbc) continue;
                            cc = (cc + nc2) % nc2;
                        }
                        const int cell = (ca * nc1 + cb) * nc2 + cc;
                        for (int s = cell_start[cell]; s < cell_start[cell + 1]; ++s) {
                            const int q = cell_list[s];
                            if (fast[q]) continue;  // includes q == p
                            pair(p, q, fx, fy, fz);
                        }
                        // fast partners binned at their current positions; q > p counts each pair once
                        auto lo = std::lower_bound(fast_binned.begin(), fast_binned.end(), std::make_pair(cell, 0));
                        for (; lo != fast_binned.end() && lo->first == cell; ++lo)
                            if (lo->second > p) pair(p, lo->second, fx, fy, fz);
                    }
                }
            }
            f[3 * p] += fx;
            f[3 * p + 1] += fy;
            f[3 * p + 2] += fz;
            fast_span[a] = {tid, out0, out.size() - out0};
        }
#pragma omp single
        for (int a = 0; a < nfast; ++a) {
            const FastSpan& sp = fast_span[a];
            const Contribution* c = thr_scratch[sp.tid].contrib.data() + sp.start;
            for (size_t k = 0; k < sp.count; ++k) {
                f[3 * c[k].q] += c[k].fx;
                f[3 * c[k].q + 1] += c[k].fy;
                f[3 * c[k].q + 2] += c[k].fz;
            }
        }
    }

    // ---- forces (called by all threads of a team) ------------------------------------
    void forces_inner() {
        if (forces_valid) return;  // shared flag, consistent after the last barrier
        const int tid = thread_id();
        // Neighbour lists.  All lists share x_ref and the image offsets, so if
        // one needs a rebuild, all are rebuilt.  The decision is taken by one
        // thread and published by the barrier at the end of the single block.
#pragma omp single
        {
            rebuild_all = false;
            for (auto& pf : pairs)
                if (pf.need_rebuild) rebuild_all = true;
            // the direct search for the fast set uses the grid of the (single)
            // pair force and needs >= 3 cells per periodic dimension
            const bool escape_ok = pairs.size() == 1 && (!pbc || (grid_nc[0] >= 3 && grid_nc[1] >= 3 && grid_nc[2] >= 3));
            if (nfast > (escape_ok ? fast_max : 0)) rebuild_all = true;
        }
        if (rebuild_all) {
            for (auto& pf : pairs) build_list_inner(pf);
#pragma omp for schedule(static)
            for (int p = 0; p < N; ++p) fast[p] = 0;
#pragma omp single
            {
                nfast = 0;
                fast_list.clear();
            }
        }
        // xw = x - img (wrapped coordinates, PBC only) is kept up to date by
        // the integrator; after a rebuild, img has changed.
        if (rebuild_all && pbc) {
#pragma omp for schedule(static)
            for (size_t c = 0; c < 3 * (size_t)N; ++c) xw[c] = x[c] - img[c];
        }
        const double* const XW = pbc ? xw.data() : x.data();

        double e = 0.0;
        const uint8_t* is_fast = nfast > 0 ? fast.data() : nullptr;
        // pair forces + external forces, owner computes; the fast set is handled below
#pragma omp for schedule(dynamic, 64)
        for (int p = 0; p < N; ++p) {
            double fx = 0, fy = 0, fz = 0, ep = 0;
            const double px = XW[3 * p], py = XW[3 * p + 1], pz = XW[3 * p + 2];
            if (!is_fast || !is_fast[p])
            for (const PairForce& pf : pairs) {
                const double rc2 = pf.cutoff * pf.cutoff;
                const Table* row = &pf.tables[(size_t)pf.type[p] * pf.ntypes];
                const bool one_type = pf.ntypes == 1;
                const uint32_t* nb = pf.nb.data();
                for (int s = pf.nb_start[p], se = pf.nb_start[p + 1]; s < se; ++s) {
                    const uint32_t ent = nb[s];
                    const int q = (int)(ent & NB_INDEX_MASK);
                    if (is_fast && is_fast[q]) continue;
                    const double* sh = shifts[ent >> NB_SHIFT_BITS];
                    const double dx = px - XW[3 * q] - sh[0], dy = py - XW[3 * q + 1] - sh[1],
                                 dz = pz - XW[3 * q + 2] - sh[2];
                    const double r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 >= rc2) continue;
                    const double r = std::sqrt(r2);
                    double en, de;
                    (one_type ? row[0] : row[pf.type[q]]).eval_in_range(r, en, de);
                    ep += en;
                    const double sc = r > 0 ? -de / r : 0.0;
                    fx += sc * dx;
                    fy += sc * dy;
                    fz += sc * dz;
                }
            }
            e += 0.5 * ep;
            f[3 * p] = fx;
            f[3 * p + 1] = fy;
            f[3 * p + 2] = fz;
        }
        if (nfast > 0) fast_set_forces(e);
        for (const SphericalConfinement& sc : spheres) {
            const int n = (int)sc.particles.size();
#pragma omp for schedule(static)
            for (int q = 0; q < n; ++q) {
                const int p = sc.particles[q];
                const double dx = x[3 * p] - sc.cx, dy = x[3 * p + 1] - sc.cy, dz = x[3 * p + 2] - sc.cz;
                const double r = std::sqrt(dx * dx + dy * dy + dz * dz + sc.tt * sc.tt);
                const double u = r - sc.aa;
                if (sc.sign * u < 0) continue;  // Lepton: step(x) = x >= 0
                const double root = std::sqrt(u * u + sc.t * sc.t);
                e += sc.kb * (root - sc.t);
                const double s = -sc.kb * u / (root * r);
                f[3 * p] += s * dx;
                f[3 * p + 1] += s * dy;
                f[3 * p + 2] += s * dz;
            }
        }
        for (const Tether& t : tethers) {
            const int n = (int)t.particles.size();
            const bool tp = t.pbc && pbc;
#pragma omp for schedule(static)
            for (int q = 0; q < n; ++q) {
                const int p = t.particles[q];
                double dx = x[3 * p] - t.x0[3 * q], dy = x[3 * p + 1] - t.x0[3 * q + 1],
                       dz = x[3 * p + 2] - t.x0[3 * q + 2];
                if (tp) min_image(dx, dy, dz);
                e += t.k[0] * dx * dx + t.k[1] * dy * dy + t.k[2] * dz * dz;
                f[3 * p] -= 2 * t.k[0] * dx;
                f[3 * p + 1] -= 2 * t.k[1] * dy;
                f[3 * p + 2] -= 2 * t.k[2] * dz;
            }
        }
        // bonded terms into the contribution buffer
        for (const BondSet& b : bonds) {
            const int n = (int)b.i.size();
            const bool bp = b.pbc && pbc;
            double* C = contrib.data() + 3 * b.contrib_offset;
#pragma omp for schedule(static) nowait
            for (int q = 0; q < n; ++q) {
                const int i = b.i[q], j = b.j[q];
                double dx = x[3 * i] - x[3 * j], dy = x[3 * i + 1] - x[3 * j + 1], dz = x[3 * i + 2] - x[3 * j + 2];
                if (bp) min_image(dx, dy, dz);
                const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                double en, de;
                if (b.tabulated) {
                    b.tables[b.tab[q]].eval(r, en, de);
                } else {
                    const double ext = r - b.r0[q];
                    de = b.k[q] * ext;
                    en = 0.5 * de * ext;
                }
                e += en;
                const double s = r > 0 ? -de / r : 0.0;
                double* c = C + 6 * (size_t)q;
                c[0] = s * dx;
                c[1] = s * dy;
                c[2] = s * dz;
                c[3] = -c[0];
                c[4] = -c[1];
                c[5] = -c[2];
            }
        }
        for (const AngleSet& a : angles) {
            const int n = (int)a.i.size();
            const bool ap = a.pbc && pbc;
            double* C = contrib.data() + 3 * a.contrib_offset;
#pragma omp for schedule(static) nowait
            for (int q = 0; q < n; ++q) {
                const int i = a.i[q], j = a.j[q], k = a.k[q];
                double ax = x[3 * i] - x[3 * j], ay = x[3 * i + 1] - x[3 * j + 1], az = x[3 * i + 2] - x[3 * j + 2];
                double bx = x[3 * k] - x[3 * j], by = x[3 * k + 1] - x[3 * j + 1], bz = x[3 * k + 2] - x[3 * j + 2];
                if (ap) {
                    min_image(ax, ay, az);
                    min_image(bx, by, bz);
                }
                double* c = C + 9 * (size_t)q;
                const AngleTable& at = a.tables[a.tab[q]];
                const double aa2 = ax * ax + ay * ay + az * az, bb2 = bx * bx + by * by + bz * bz;
                const double dot = ax * bx + ay * by + az * bz;
                double en, de;
                if (aa2 == 0 || bb2 == 0) {
                    at.t.eval(0.0, en, de);
                    e += en;
                    for (int z = 0; z < 9; ++z) c[z] = 0.0;
                    continue;
                }
                const double inv_ab = 1.0 / std::sqrt(aa2 * bb2);
                double cs = dot * inv_ab;
                cs = cs > 1.0 ? 1.0 : (cs < -1.0 ? -1.0 : cs);
                if (cs >= at.c_lo && cs <= at.c_hi) {
                    at.tc.eval(cs, en, de);  // de = dE/dcos
                    e += en;
                    // f_i = -dE/dc * (b/(|a||b|) - c a/|a|^2), f_k likewise
                    const double ca = cs / aa2, cb = cs / bb2;
                    const double fix = -de * (bx * inv_ab - ca * ax), fiy = -de * (by * inv_ab - ca * ay),
                                 fiz = -de * (bz * inv_ab - ca * az);
                    const double fkx = -de * (ax * inv_ab - cb * bx), fky = -de * (ay * inv_ab - cb * by),
                                 fkz = -de * (az * inv_ab - cb * bz);
                    c[0] = fix;
                    c[1] = fiy;
                    c[2] = fiz;
                    c[3] = -fix - fkx;
                    c[4] = -fiy - fky;
                    c[5] = -fiz - fkz;
                    c[6] = fkx;
                    c[7] = fky;
                    c[8] = fkz;
                } else {
                    const double nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
                    const double nn = std::sqrt(nx * nx + ny * ny + nz * nz);
                    const double theta = std::atan2(nn, dot);
                    at.t.eval(theta, en, de);  // de = dE/dtheta
                    e += en;
                    if (nn < 1e-12 * std::sqrt(aa2 * bb2)) {
                        for (int z = 0; z < 9; ++z) c[z] = 0.0;
                        continue;
                    }
                    // f_i = dE/dtheta (n x a) / (|n| |a|^2), f_k = dE/dtheta (b x n) / (|n| |b|^2)
                    const double si = de / (nn * aa2), sk = de / (nn * bb2);
                    const double fix = si * (ny * az - nz * ay), fiy = si * (nz * ax - nx * az),
                                 fiz = si * (nx * ay - ny * ax);
                    const double fkx = sk * (by * nz - bz * ny), fky = sk * (bz * nx - bx * nz),
                                 fkz = sk * (bx * ny - by * nx);
                    c[0] = fix;
                    c[1] = fiy;
                    c[2] = fiz;
                    c[3] = -fix - fkx;
                    c[4] = -fiy - fky;
                    c[5] = -fiz - fkz;
                    c[6] = fkx;
                    c[7] = fky;
                    c[8] = fkz;
                }
            }
        }
        thr[(size_t)PAD * tid] = e;
#pragma omp barrier
        // Gather the bonded contributions; this is the last update of f, so the
        // variable Langevin error sum (sum of |f/m|^2, see integrate_inner) is
        // accumulated here, in fixed blocks of particles, so that it does not
        // depend on the number of threads.
        const bool vl = integrator == VARIABLE_LANGEVIN;
        const int nblk = (N + SUM_BLOCK - 1) / SUM_BLOCK;
#pragma omp for schedule(static)
        for (int b = 0; b < nblk; ++b) {
            double err = 0.0;
            for (int p = b * SUM_BLOCK, pe = std::min(N, p + SUM_BLOCK); p < pe; ++p) {
                double fx = 0, fy = 0, fz = 0;
                for (int s = gather_start[p]; s < gather_start[p + 1]; ++s) {
                    const double* c = &contrib[3 * (size_t)gather_idx[s]];
                    fx += c[0];
                    fy += c[1];
                    fz += c[2];
                }
                fx += f[3 * p];
                fy += f[3 * p + 1];
                fz += f[3 * p + 2];
                f[3 * p] = fx;
                f[3 * p + 1] = fy;
                f[3 * p + 2] = fz;
                if (vl) {
                    const double im = inv_mass[p], ax = im * fx, ay = im * fy, az = im * fz;
                    err += ax * ax;
                    err += ay * ay;
                    err += az * az;
                }
            }
            block_sum[b] = err;
        }
#pragma omp single
        {
            epot = thr_sum();
            if (!std::isfinite(epot)) set_error("non-finite potential energy (simulation exploded?)");
            if (vl) {
                double err = 0.0;
                for (int b = 0; b < nblk; ++b) err += block_sum[b];
                vl_err = err;
                if (stepping) {
                    select_step_size();
                    dt_selected = true;
                }
            }
            forces_valid = true;
        }  // implicit barrier
    }

    // ReferenceVariableStochasticDynamics::updatePart1: step size from the error
    // sum over all coordinates of (f/m)^2 (vl_err, computed with the forces).
    void select_step_size() {
        double er = std::sqrt(vl_err / (3.0 * N));
        double nd = er > 0 ? std::sqrt(error_tol_ / er) : max_dt;
        if (dt_ > 0.0) nd = std::min(nd, 2.0 * dt_);
        if (nd > dt_ && nd < 1.2 * dt_) nd = dt_;
        if (nd > max_dt) nd = max_dt;
        if (!std::isfinite(nd) || nd <= 0) {
            set_error("variable Langevin: invalid step size (simulation exploded?)");
            nd = dt_ > 0 ? dt_ : 1e-6;
        }
        dt_ = nd;
    }

    // ---- integration (called by all threads of a team) ------------------------------------
    void integrate_inner() {
        const int tid = thread_id();
        if (integrator == VARIABLE_LANGEVIN && !dt_selected) {
            // forces were already valid when the step started (e.g. after getState).
            // dt_selected is only written in single blocks of forces_inner and
            // below, so all threads take this branch alike.
#pragma omp single
            select_step_size();
        }
        const double dt = dt_;
        const double kT = BOLTZ * T;
        const double vscale = std::exp(-dt * gamma);
        const double noisescale = std::sqrt(kT * (1 - vscale * vscale));
        const uint64_t base = mix64(seed ^ mix64((uint64_t)nsteps));
        std::vector<int>& my_fast = thr_scratch[tid].fast;
        my_fast.clear();
        bool bad = false;

#pragma omp for schedule(static)
        for (int p = 0; p < N; ++p) {
            const double im = inv_mass[p];
            if (im == 0) continue;
            const double sim = std::sqrt(im);
            double d2 = 0.0;
            for (int d = 0; d < 3; ++d) {
                const size_t c = 3 * (size_t)p + d;
                const double g = znormal(base + ((uint64_t)c << 24));
                if (integrator == BROWNIAN) {
                    // ReferenceBrownianDynamics
                    const double dx = dt / gamma * im * f[c] + std::sqrt(2.0 * kT * dt / gamma) * sim * g;
                    x[c] += dx;
                    v[c] = dx / dt;
                } else {
                    // ReferenceLangevinMiddleDynamics (OpenMM 8: also LangevinIntegrator and,
                    // with the step size chosen above, VariableLangevinIntegrator)
                    double vv = v[c] + dt * im * f[c];
                    double xx = x[c] + 0.5 * dt * vv;
                    vv = vscale * vv + noisescale * sim * g;
                    xx += 0.5 * dt * vv;
                    v[c] = vv;
                    x[c] = xx;
                }
                if (pbc) xw[c] = x[c] - img[c];
                const double dd = x[c] - x_ref[c];
                d2 += dd * dd;
            }
            if (!(d2 <= std::numeric_limits<double>::max())) bad = true;  // NaN or inf
            const bool fp = d2 > fast_lim2;
            fast[p] = fp;
            if (fp) my_fast.push_back(p);
        }
        thr[(size_t)PAD * tid] = bad ? 1.0 : 0.0;
#pragma omp barrier
#pragma omp single
        {
            if (thr_sum() != 0.0) set_error("non-finite coordinates (simulation exploded?)");
            fast_list.clear();
            for (const ThreadScratch& ts : thr_scratch) fast_list.insert(fast_list.end(), ts.fast.begin(), ts.fast.end());
            nfast = (int)fast_list.size();
            time_ += dt;
            nsteps++;
            forces_valid = false;
            dt_selected = false;
        }  // implicit barrier
    }
};

}  // namespace pcn
