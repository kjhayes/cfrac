
#define FLOAT double

#define ITER_STEP 100

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>

#include <ncurses.h>

#define USE_OPENMP
#ifdef USE_OPENMP
#include <omp.h>
#endif

/*
 * Complex Number Implementation
 */
typedef struct complex {
    FLOAT real;
    FLOAT imag;
} complex_t;

static inline complex_t
complex_add(complex_t a, complex_t b)
{
    complex_t sum = {
        .real = a.real + b.real,
        .imag = a.imag + b.imag,
    };
    return sum;
}

static inline complex_t
complex_sub(complex_t a, complex_t b)
{
    complex_t sum = {
        .real = a.real - b.real,
        .imag = a.imag - b.imag,
    };
    return sum;
}

static inline complex_t
complex_mult(complex_t a, complex_t b)
{
    complex_t prod = {
        .real = (a.real * b.real) - (a.imag * b.imag),
        .imag = (a.real*b.imag) + (a.imag*b.real),
    };
    return prod;
}

static inline complex_t
complex_div(complex_t num, complex_t den)
{
    FLOAT divisor = (den.real*den.real) + (den.imag*den.imag);
    complex_t quot = {
        .real = ((num.real*den.real) + (num.imag*den.imag)) / divisor,
        .imag = ((num.imag*den.real) - (num.real*den.imag)) / divisor,
    };
    return quot;
}

static inline FLOAT
complex_sqr_mag(complex_t c) {
    return (c.real*c.real) + (c.imag*c.imag);
}

static inline complex_t
complex_real(FLOAT r)
{
    complex_t c = {
        .real = r,
        .imag = 0.0,
    };
    return c;
}

static inline complex_t
complex_conj(complex_t c)
{
    complex_t conj = {
        .real = c.real,
        .imag = -c.imag,
    };
    return conj;
}

static int
cfrac_setup_ncurses(void)
{
    initscr();
	if(has_colors() == FALSE)
	{
		fprintf(stderr, "Your terminal does not support color\n");
        abort();
	}

    cbreak();
    noecho();
    timeout(1);
    keypad(stdscr, TRUE);
    curs_set(FALSE);

	start_color();
	init_pair(0, COLOR_WHITE, COLOR_BLACK);
	init_pair(1, COLOR_RED, COLOR_BLACK);
	init_pair(2, COLOR_GREEN, COLOR_BLACK);
	init_pair(3, COLOR_BLUE, COLOR_BLACK);
	init_pair(4, COLOR_YELLOW, COLOR_BLACK);
    return 0;
}

static void
cfrac_teardown_ncurses(void)
{
    endwin();
}

// Two array of complex values
struct complex_plane {
    size_t res_x, res_y;

    complex_t center;
    FLOAT width;
    FLOAT height;

    complex_t values[];
};

static struct complex_plane *
complex_plane_create(
        size_t res_x,
        size_t res_y)
{
    size_t size = sizeof(struct complex_plane) + (res_x*res_y*sizeof(complex_t));
    struct complex_plane *plane = malloc(size);
    if(plane == NULL) {
        return NULL;
    }
    plane->res_x = res_x;
    plane->res_y = res_y;
    plane->center = complex_real(0.0);
    plane->width = 1.0;
    plane->height = 1.0;
    return plane;
}

static int
complex_plane_destroy(
        struct complex_plane *plane)
{
    free(plane);
    return 0;
}

static inline complex_t *
complex_plane_ptr(
        struct complex_plane *plane,
        size_t x,
        size_t y)
{
    return &plane->values[x + (y*plane->res_x)];
}

static inline complex_t
complex_plane_loc(
        struct complex_plane *plane,
        size_t x,
        size_t y)
{
    FLOAT real_base = plane->center.real - (plane->width/2.0);
    FLOAT imag_base = plane->center.imag + (plane->height/2.0);
    FLOAT real_step =  (plane->width/plane->res_x);
    FLOAT imag_step = -(plane->height/plane->res_y);

    complex_t loc = {
        .real = real_base + (x*real_step),
        .imag = imag_base + (y*imag_step),
    };

    return loc;
}

static void
center_complex_plane(
        struct complex_plane *plane,
        complex_t center,
        FLOAT width,
        FLOAT height)
{
    plane->center = center;
    plane->width = width;
    plane->height = height;

    #pragma omp parallel for
    for(size_t y = 0; y < plane->res_y; y++) {
        for(size_t x = 0; x < plane->res_x; x++) {
            complex_t *c = complex_plane_ptr(plane, x, y);
            *c = complex_plane_loc(plane, x, y);
        }
    }
}

static void
apply_function_to_complex_plane(
        struct complex_plane *plane,
        size_t iter,
        complex_t(*f)(complex_t, complex_t, size_t))
{
    #pragma omp parallel for
    for(size_t y = 0; y < plane->res_y; y++) {
        for(size_t x = 0; x < plane->res_x; x++) {
            complex_t *c = complex_plane_ptr(plane, x, y);
            complex_t l = complex_plane_loc(plane, x, y);
            *c = (*f)(*c, l, iter);
        }
    }
}

struct fractal {
    enum {
        RENDER_MODE_QUADRANT,
        RENDER_MODE_DIVERGING,
    } render_mode;
    complex_t (*dynamical_system)(complex_t,complex_t,size_t);
    size_t starting_target_iterations;
    complex_t starting_center;
    FLOAT starting_width;
    FLOAT starting_height;
    const char *name;
    const char *opt_string;
};

/* z^3 - 1 Newton's Fractal */
static complex_t dynamical_system_newton_z3_1(complex_t z, complex_t loc, size_t iter)
{
    // z - ((z^3 - 1)/(3z^2))
    complex_t z_sqr = complex_mult(z, z);
    complex_t z_cubed = complex_mult(z_sqr, z);
    complex_t f_z = complex_sub(z_cubed, complex_real(-1.0));
    complex_t f_prime_z = complex_mult(z_sqr, complex_real(3.0));
    complex_t quot = complex_div(f_z, f_prime_z);
    complex_t final = complex_sub(z, quot);
    return final;
}

static struct fractal
z3_min_1_newton_fractal = {
    .render_mode = RENDER_MODE_QUADRANT,
    .dynamical_system = dynamical_system_newton_z3_1,
    .starting_target_iterations = 1000,
    .starting_center = {.real=0.0,.imag=0.0},
    .starting_width = 3.0,
    .starting_height = 3.0,
    .name = "Newton's Fractal on z^3 - 1",
};

/* Mandelbrot Set */
static complex_t dynamical_system_mandelbrot(complex_t z, complex_t loc, size_t iter)
{
    // z^2 + c
    FLOAT m = complex_sqr_mag(z);
    if(m > 2.0) {
        if(iter > 2) {
            return z;
        } else if(iter < 2) {
            return complex_real(2.1);
        }
    }

    complex_t z_sqr = complex_mult(z, z);
    complex_t final = complex_add(z_sqr, loc);

    m = complex_sqr_mag(final);
    if(m > 2.0) {
        if(iter > 2) {
            return complex_real(iter);
        }
    }
    return final;
}

static struct fractal
mandelbrot_set_fractal = {
    .render_mode = RENDER_MODE_DIVERGING,
    .dynamical_system = dynamical_system_mandelbrot,
    .starting_target_iterations = 1000,
    .starting_center = {.real=-0.5,.imag=0.0},
    .starting_width = 2.0,
    .starting_height = 2.0,
    .name = "Mandelbrot Set",
};

static struct fractal *fractals[] = {
    &z3_min_1_newton_fractal,
    &mandelbrot_set_fractal,
};

#define NUM_FRACTALS (sizeof(fractals) / sizeof(struct fractal *))

#define ALERT_BUFLEN 256
#define ALERT_SEC 3

struct cfrac_state {
    int running;
    int need_recompute;

    complex_t center;
    FLOAT width;
    FLOAT height;

    size_t res_x;
    size_t res_y;

    struct complex_plane *plane;

    size_t fractal_index;

    size_t target_iterations;
    size_t current_iteration;
    ssize_t current_rendered_iteration;

    char alert_msg_buffer[ALERT_BUFLEN];
    time_t alert_msg_end;
};

#define cfrac_alert(state, fmt, ...) \
do {\
    snprintf(state->alert_msg_buffer, ALERT_BUFLEN, fmt, ##__VA_ARGS__);\
    state->alert_msg_buffer[ALERT_BUFLEN-1] = '\0';\
    state->alert_msg_end = time(NULL) + ALERT_SEC; \
} while(0)

static void
resize_step(struct cfrac_state *state)
{
    int term_x,term_y;
    getmaxyx(stdscr, term_y, term_x);

    if(state->plane == NULL
    || state->res_x != term_x
    || state->res_y != term_y
    || state->need_recompute)
    {
        if(state->res_x != term_x || state->res_y != term_y) {
            if(state->plane != NULL) {
                complex_plane_destroy(state->plane);
                state->plane = NULL;
            }
            state->res_x = term_x;
            state->res_y = term_y;
            state->plane = complex_plane_create(state->res_x, state->res_y);
            if(state->plane == NULL) {
                fprintf(stderr, "Failed to reallocate backing buffer!\n");
                abort();
            }
        }

        center_complex_plane(state->plane, state->center, state->width, state->height);

        state->current_iteration = 0;
        state->current_rendered_iteration = -1;

        state->need_recompute = 0;
    }
}

static void
compute_step(struct cfrac_state *state)
{
    struct fractal *fractal = fractals[state->fractal_index];

    if(state->plane == NULL || fractal == NULL) {
        return;
    }

    time_t start = time(NULL);

    while(1) {
        if(state->current_iteration < state->target_iterations)
        {
            apply_function_to_complex_plane(state->plane, state->current_iteration, fractal->dynamical_system);
            state->current_iteration++;
            if(start != time(NULL)) {
                break;
            }
        } else {
            break;
        }
    }
}

static inline int 
render_point_quadrant(
        struct cfrac_state *state,
        complex_t val)
{
    int ch;
    if(val.real >= 0.0) {
        if(val.imag >= 0.0) {
            ch = '1' | COLOR_PAIR(1);
        } else {
            ch = '4' | COLOR_PAIR(2);
        }
    } else {
        if(val.imag >= 0.0) {
            ch = '2' | COLOR_PAIR(3);
        } else {
            ch = '3' | COLOR_PAIR(4);
        }
    }

    return ch;
}

static inline int 
render_point_diverging(
        struct cfrac_state *state,
        complex_t val)
{
    const static struct {
        FLOAT size;
        int ch;
    } regions[] = {
        { .size = 2.0, .ch = ' ' | COLOR_PAIR(0), },
        { .size = 1000.0, .ch = '#' | COLOR_PAIR(0), },
        { .size = 10000.0, .ch = '#' | COLOR_PAIR(2), },
        { .size = 100000.0, .ch = '#' | COLOR_PAIR(4), },
    };
    size_t num_regions = sizeof(regions) / sizeof(regions[0]);

    FLOAT m = complex_sqr_mag(val);
    FLOAT cur_cutoff = 0.0;
    for(size_t i = 0; i < num_regions; i++) {
        cur_cutoff += regions[i].size;
        if(m < cur_cutoff) {
            return regions[i].ch;
        }
    }

    return '#' | COLOR_PAIR(1);
}

static void
render_step(struct cfrac_state *state)
{
    if(state->plane == NULL) {
        return;
    }

    if(state->current_iteration == state->current_rendered_iteration) {
        return;
    }

    struct fractal *fractal = fractals[state->fractal_index];

    for(size_t y = 0; y < state->res_y; y++) {
        for(size_t x = 0; x < state->res_x; x++) {
            complex_t *c = complex_plane_ptr(state->plane, x, y);
            complex_t l = complex_plane_loc(state->plane, x, y);
            int ch;
            switch(fractal->render_mode) {
                case RENDER_MODE_QUADRANT:
                    ch = render_point_quadrant(state, *c);
                    break;
                case RENDER_MODE_DIVERGING:
                    ch = render_point_diverging(state, *c);
                    break;
                default:
                    ch = '?';
                    break;
            }
            mvaddch(y,x,ch);
        }
    }
    move(state->res_y-1,0);
    time_t cur_time = time(NULL);
    if(cur_time > state->alert_msg_end) {
        printw("%s (Num Iter. %lu/%lu) (Centered on %le+%lei) (width=%le,height=%le)",
            fractal->name ? fractal->name : "",
            state->current_iteration, state->target_iterations,
            state->center.real, state->center.imag,
            state->width, state->height
            );
    } else {
        printw("%s", state->alert_msg_buffer);
    }
    refresh();
}

static void
input_step(struct cfrac_state *state)
{
#define MAX_BATCH 64

    int batch_size = 0;

    FLOAT real_step = (state->width / state->res_x);
    FLOAT imag_step = (state->height / state->res_y);


    while(batch_size < MAX_BATCH) {
        int ch = getch();
        batch_size++;
        switch(ch) {

            case 'n':
            case 'N':
                state->fractal_index++;
                if(state->fractal_index >= NUM_FRACTALS) {
                    state->fractal_index = 0;
                }
                state->target_iterations = fractals[state->fractal_index]->starting_target_iterations,
                state->center = fractals[state->fractal_index]->starting_center,
                state->width = fractals[state->fractal_index]->starting_width,
                state->height = fractals[state->fractal_index]->starting_height,
                state->need_recompute = 1;
                break;
            case 'a':
            case 'A':
            case 'h':
            case 'H':
            case KEY_LEFT:
                state->center.real -= real_step;
                state->need_recompute = 1;
                break;

            case 'd':
            case 'D':
            case 'l':
            case 'L':
            case KEY_RIGHT:
                state->center.real += real_step;
                state->need_recompute = 1;
                break;

            case 'w':
            case 'W':
            case 'k':
            case 'K':
            case KEY_UP:
                state->center.imag += imag_step;
                state->need_recompute = 1;
                break;

            case 's':
            case 'S':
            case 'j':
            case 'J':
            case KEY_DOWN:
                state->center.imag -= imag_step;
                state->need_recompute = 1;
                break;

            case 'i':
                state->target_iterations -= ITER_STEP;
                state->need_recompute = 1;
                break;
            case 'I':
                state->target_iterations += ITER_STEP;
                break;

            case '+':
                state->width *= 0.9;
                state->height *= 0.9;
                state->need_recompute = 1;
                break;
            case '-':
                state->width *= (1.0/0.9);
                state->height *= (1.0/0.9);
                state->need_recompute = 1;
                break;

            case ERR:
                // Timed out
                batch_size = MAX_BATCH;
                break;

            case 'r':
            case 'R':
                state->need_recompute = 1;
                batch_size = MAX_BATCH;
                break;
            case 'q':
            case 'Q':
            case KEY_END:
            case KEY_EOL:
                state->running = 0;
                batch_size = MAX_BATCH;
                break;
            case '?':
                cfrac_alert(state, "Move[WASD,Arrow Keys,VIM Motions] Zoom[+/-] Next Fractal[N] Change Iterations [i/I]");
                break;
            default:
                cfrac_alert(state, "Unrecognized Input! (Press '?' to See Current Options)");
                break;
        }
    }
}

int
main(int argc,
     const char **argv)
{
    int res;

    // Setup ncurses and register the atexit teardown for ncurses
    res = cfrac_setup_ncurses();
    if(res) {
        fprintf(stderr, "Failed to setup ncurses!\n");
        return -1;
    }
    if(atexit(cfrac_teardown_ncurses) != 0) {
        perror("atexit");
        return -1;
    }
    // We should now be safe to call any ncurses functions

    struct cfrac_state state = {
        .running = 1,
        .need_recompute = 1,
        .plane = NULL,
        .current_iteration = 0,
        .current_rendered_iteration = -1,
        .alert_msg_end = 0,

        .fractal_index = 0,

        // Initial Parameters
        .target_iterations = fractals[0]->starting_target_iterations,
        .center = fractals[0]->starting_center,
        .width = fractals[0]->starting_width,
        .height = fractals[0]->starting_height,
    };
    while(state.running) {
        resize_step(&state);
        compute_step(&state);
        render_step(&state);
        input_step(&state);
    }

    return 0;
}

