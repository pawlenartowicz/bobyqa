// oracle/driver.c — PRIMA BOBYQA capture driver.
//
// Solves one registry problem with prima_minimize() and writes the run to
// stdout in the "bobyqa golden v1" body format; capture.sh prepends the three
// `#` metadata lines (the driver cannot know the pin hash / compiler version).
// See the pinned clone's own C example:
// oracle/prima/c/examples/bobyqa/bobyqa_example.c.
//
// Usage: driver <problem> [--npt N] [--rhobeg X] [--rhoend X] [--maxfun N]
//                  [--time SECONDS]
//
// With `--time S` the driver switches to timing mode: it repeats the same solve
// for at least S seconds and prints one summary line instead of the golden body.
// Nothing about the solve itself changes -- only the printf calls around it --
// so the captured goldens are unaffected (verified by diffing a recapture).

#include "prima/prima.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_N 16 /* registry dimension cap */

typedef double (*objective_fn)(int n, const double *x);

/* Objectives — MUST stay arithmetically bit-identical to the Rust renderings
   in bobyqa/tests/parity_prima.rs: same operations, same order, and compiled
   with -ffp-contract=off so no FMA contraction sneaks in (full-trajectory
   parity dies on a last-bit f difference). */

static double sphere(int n, const double *x)
{
    double f = 0.0;
    for (int i = 0; i < n; ++i)
        f += x[i] * x[i];
    return f;
}

static double rosenbrock(int n, const double *x)
{
    double f = 0.0;
    for (int i = 0; i + 1 < n; ++i) {
        const double a = x[i + 1] - x[i] * x[i];
        const double b = 1.0 - x[i];
        f += 100.0 * (a * a) + b * b;
    }
    return f;
}

static double booth(int n, const double *x)
{
    (void)n;
    const double a = x[0] + 2.0 * x[1] - 7.0;
    const double b = 2.0 * x[0] + x[1] - 5.0;
    return a * a + b * b;
}

static double beale(int n, const double *x)
{
    (void)n;
    const double y = x[1];
    const double a = 1.5 - x[0] + x[0] * y;
    const double b = 2.25 - x[0] + x[0] * (y * y);
    const double c = 2.625 - x[0] + x[0] * ((y * y) * y);
    return a * a + b * b + c * c;
}

static double powell_singular(int n, const double *x)
{
    (void)n;
    const double a = x[0] + 10.0 * x[1];
    const double b = x[2] - x[3];
    const double c = x[1] - 2.0 * x[2];
    const double d = x[0] - x[3];
    const double c2 = c * c;
    const double d2 = d * d;
    return a * a + 5.0 * (b * b) + c2 * c2 + 10.0 * (d2 * d2);
}

static double nansphere(int n, const double *x)
{
    if (x[0] < 0.0)
        return NAN;
    return sphere(n, x);
}

/* One-dimensional objectives (n = 1, where npt = 3 is the only legal value). */
static double quad1(int n, const double *x)
{
    (void)n;
    const double d = x[0] - 0.7;
    return 4.0 * (d * d) + 1.0;
}

static double quad1_shift(int n, const double *x)
{
    (void)n;
    const double d = x[0] + 2.0;
    return d * d;
}

static double quartic1(int n, const double *x)
{
    (void)n;
    const double x2 = x[0] * x[0];
    return x2 * x2 - 3.0 * x2 + x[0];
}

static double reml1(int n, const double *x)
{
    (void)n;
    const double d = 1.0 + 20.0 * (x[0] * x[0]);
    return 150.0 * log(d) + 2980.0 * log(1.0 + 64.0 / d);
}

typedef struct {
    const char *name;
    int n;
    objective_fn fun;
    double x0[MAX_N];
    double lower[MAX_N];
    double upper[MAX_N];
    double rhobeg;
    double rhoend;
    int maxfun;
} problem_t;

static const problem_t REGISTRY[] = {
    {"sphere",     2, sphere,     {1.0, 2.0},  {-5.0, -5.0}, {5.0, 5.0},   0.5, 1e-6, 500},
    {"rosenbrock", 2, rosenbrock, {-1.2, 1.0}, {-5.0, -5.0}, {10.0, 10.0}, 0.5, 1e-6, 500},
    {"booth",        2, booth,      {0.0, 0.0},  {-10.0, -10.0}, {10.0, 2.5}, 0.5, 1e-6, 500},
    {"rosenbrock10", 10, rosenbrock,
     {-1.2, 1.0, -1.2, 1.0, -1.2, 1.0, -1.2, 1.0, -1.2, 1.0},
     {-5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0},
     {10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0}, 0.5, 1e-6, 2000},
    {"beale",           2, beale,           {1.0, 1.0},  {-4.5, -4.5}, {4.5, 4.5}, 0.5, 1e-6, 500},
    {"powell_singular", 4, powell_singular, {3.0, -1.0, 0.0, 1.0}, {-4.0, -4.0, -4.0, -4.0}, {5.0, 5.0, 5.0, 5.0}, 0.5, 1e-6, 2000},
    {"sphere_onbound",  2, sphere,          {1.2, 0.3},  {1.0, -5.0},  {6.0, 5.0}, 0.5, 1e-6, 500},
    {"sphere_tight",    2, sphere,          {0.55, -0.3}, {-0.6, -0.6}, {0.6, 0.6}, 0.5, 1e-6, 500},
    {"nansphere",       2, nansphere,       {0.5, 2.0},  {-5.0, -5.0}, {5.0, 5.0}, 0.5, 1e-6, 500},
    {"quad1",           1, quad1,           {3.0},       {-10.0},      {10.0},     0.5, 1e-8, 500},
    {"quad1_lower",     1, quad1_shift,     {3.0},       {0.0},        {5.0},      0.5, 1e-8, 500},
    {"quartic1",        1, quartic1,        {2.5},       {-3.0},       {3.0},      0.5, 1e-8, 500},
    {"reml1",           1, reml1,           {1.0},       {0.0},        {1000.0},   0.5, 1e-6, 500},
};

/* Single-token names for the `final` line. prima_get_rc_string() returns
   sentences with spaces (verified at v0.7.2), which would break the
   whitespace-tokenised golden format. */
static const char *rc_name(int rc)
{
    switch (rc) {
    case PRIMA_SMALL_TR_RADIUS:          return "PRIMA_SMALL_TR_RADIUS";
    case PRIMA_FTARGET_ACHIEVED:         return "PRIMA_FTARGET_ACHIEVED";
    case PRIMA_TRSUBP_FAILED:            return "PRIMA_TRSUBP_FAILED";
    case PRIMA_MAXFUN_REACHED:           return "PRIMA_MAXFUN_REACHED";
    case PRIMA_MAXTR_REACHED:            return "PRIMA_MAXTR_REACHED";
    case PRIMA_NAN_INF_X:                return "PRIMA_NAN_INF_X";
    case PRIMA_NAN_INF_F:                return "PRIMA_NAN_INF_F";
    case PRIMA_NAN_INF_MODEL:            return "PRIMA_NAN_INF_MODEL";
    case PRIMA_NO_SPACE_BETWEEN_BOUNDS:  return "PRIMA_NO_SPACE_BETWEEN_BOUNDS";
    case PRIMA_DAMAGING_ROUNDING:        return "PRIMA_DAMAGING_ROUNDING";
    case PRIMA_INVALID_INPUT:            return "PRIMA_INVALID_INPUT";
    case PRIMA_ASSERTION_FAILS:          return "PRIMA_ASSERTION_FAILS";
    case PRIMA_VALIDATION_FAILS:         return "PRIMA_VALIDATION_FAILS";
    case PRIMA_MEMORY_ALLOCATION_FAILS:  return "PRIMA_MEMORY_ALLOCATION_FAILS";
    case PRIMA_ZERO_LINEAR_CONSTRAINT:   return "PRIMA_ZERO_LINEAR_CONSTRAINT";
    case PRIMA_CALLBACK_TERMINATE:       return "PRIMA_CALLBACK_TERMINATE";
    case PRIMA_NULL_OPTIONS:             return "PRIMA_NULL_OPTIONS";
    case PRIMA_NULL_PROBLEM:             return "PRIMA_NULL_PROBLEM";
    case PRIMA_NULL_X0:                  return "PRIMA_NULL_X0";
    case PRIMA_NULL_RESULT:              return "PRIMA_NULL_RESULT";
    case PRIMA_NULL_FUNCTION:            return "PRIMA_NULL_FUNCTION";
    case PRIMA_RESULT_INITIALIZED:       return "PRIMA_RESULT_INITIALIZED";
    default:                             return "PRIMA_UNKNOWN_RC";
    }
}

/* Timing mode suppresses every printf, including calfun's: the I/O would
   otherwise dominate the measurement (thousands of formatted lines per solve). */
static int g_quiet = 0;

static double monotonic_seconds(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* PRIMA evaluation callback: compute f and log the evaluation in call order. */
static void calfun(const double x[], double *f, const void *data)
{
    const problem_t *p = (const problem_t *)data;
    *f = p->fun(p->n, x);
    if (g_quiet)
        return;
    printf("eval");
    for (int i = 0; i < p->n; ++i)
        printf(" %.17g", x[i]);
    printf(" %.17g\n", *f);
}

static void print_vec(const char *label, const double *v, int n)
{
    printf("%s", label);
    for (int i = 0; i < n; ++i)
        printf(" %.17g", v[i]);
    printf("\n");
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <problem> [--npt N] [--rhobeg X] [--rhoend X] [--maxfun N]"
                " [--time SECONDS]\n",
                argv[0]);
        return 2;
    }

    const problem_t *p = NULL;
    for (size_t i = 0; i < sizeof REGISTRY / sizeof REGISTRY[0]; ++i) {
        if (strcmp(argv[1], REGISTRY[i].name) == 0)
            p = &REGISTRY[i];
    }
    if (p == NULL) {
        fprintf(stderr, "unknown problem '%s'\n", argv[1]);
        return 2;
    }

    int npt = 2 * p->n + 1; /* Powell's recommendation */
    double rhobeg = p->rhobeg;
    double rhoend = p->rhoend;
    int maxfun = p->maxfun;
    double time_budget = 0.0; /* > 0 switches on timing mode */
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc) {
            fprintf(stderr, "flag '%s' needs a value\n", argv[i]);
            return 2;
        }
        if (strcmp(argv[i], "--npt") == 0)
            npt = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--rhobeg") == 0)
            rhobeg = strtod(argv[i + 1], NULL);
        else if (strcmp(argv[i], "--rhoend") == 0)
            rhoend = strtod(argv[i + 1], NULL);
        else if (strcmp(argv[i], "--maxfun") == 0)
            maxfun = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--time") == 0)
            time_budget = strtod(argv[i + 1], NULL);
        else {
            fprintf(stderr, "unknown flag '%s'\n", argv[i]);
            return 2;
        }
    }

    g_quiet = (time_budget > 0.0);

    if (!g_quiet) {
        printf("problem %s\n", p->name);
        printf("n %d\n", p->n);
        printf("npt %d\n", npt);
        printf("rho_begin %.17g\n", rhobeg);
        printf("rho_end %.17g\n", rhoend);
        printf("max_fun %d\n", maxfun);
        print_vec("x0", p->x0, p->n);
        print_vec("lower", p->lower, p->n);
        print_vec("upper", p->upper, p->n);
    }

    double x0[MAX_N], xl[MAX_N], xu[MAX_N];
    memcpy(x0, p->x0, sizeof x0);
    memcpy(xl, p->lower, sizeof xl);
    memcpy(xu, p->upper, sizeof xu);

    prima_problem_t problem;
    prima_init_problem(&problem, p->n);
    problem.calfun = &calfun;
    problem.x0 = x0;
    problem.xl = xl;
    problem.xu = xu;

    prima_options_t options;
    prima_init_options(&options);
    options.npt = npt;
    options.rhobeg = rhobeg;
    options.rhoend = rhoend;
    options.maxfun = maxfun;
    options.ftarget = -INFINITY;
    options.iprint = PRIMA_MSG_NONE;
    options.data = (void *)p; /* reaches calfun's `data`, as before */

    if (g_quiet) {
        /* Repeat the identical solve; report the minimum. Timing noise is
           one-sided -- interference only ever slows a run -- so the min is the
           cleanest estimate of the cost, and the first (cold) pass is dropped. */
        prima_result_t r;
        memcpy(x0, p->x0, sizeof x0);
        prima_minimize(PRIMA_BOBYQA, problem, options, &r);
        prima_free_result(&r);

        double best = 1e300, total = 0.0;
        int reps = 0;
        double last_f = 0.0;
        int last_nf = 0;
        while (total < time_budget && reps < 1000000) {
            memcpy(x0, p->x0, sizeof x0);
            const double t0 = monotonic_seconds();
            prima_minimize(PRIMA_BOBYQA, problem, options, &r);
            const double elapsed = monotonic_seconds() - t0;
            last_f = r.f;
            last_nf = r.nf;
            prima_free_result(&r);
            if (elapsed < best)
                best = elapsed;
            total += elapsed;
            ++reps;
        }
        printf("time prima %s n %d npt %d reps %d min_ns %.0f f %.17g nf %d\n",
               p->name, p->n, npt, reps, best * 1e9, last_f, last_nf);
        return 0;
    }

    prima_result_t result;
    const prima_rc_t rc = prima_minimize(PRIMA_BOBYQA, problem, options, &result);
    if ((int)rc >= 100) { /* input/validation failure: result.x is not meaningful */
        fprintf(stderr, "prima_minimize failed: rc %d %s\n", (int)rc, rc_name((int)rc));
        return 3;
    }

    printf("final");
    for (int i = 0; i < p->n; ++i)
        printf(" %.17g", result.x[i]);
    printf(" %.17g %d %d %s\n", result.f, result.nf, (int)result.status, rc_name((int)result.status));
    prima_free_result(&result);
    return 0;
}
