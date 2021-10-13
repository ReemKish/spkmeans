#ifndef SPKMEANS_H_
#define SPKMEANS_H_

/* ===== Data structures ============================================ */
typedef struct Matrix {
	double *items;
	int rows;
	int cols;
	enum MatType {
		MAT_INVALID,
		ROW_MAJOR,
		COL_MAJOR
	} type;
} Matrix_t;

typedef struct Vector {
	double *items;
	int size;
} Vector_t;

enum Goal {
	GOAL_ERR,
	GOAL_WAM,
	GOAL_DDG,
	GOAL_LNORM,
	GOAL_JACOBI,
	GOAL_SPK
};


extern int K, D, N;	
extern Matrix_t dp_mat;

/* ===== Function prototypes ======================================== */
/* ----- Matrix data structure ---------------------------- */
Matrix_t matcreate(int rows, int cols, enum MatType type);
void matset(Matrix_t mat, int row, int col, double val);
double matget(Matrix_t mat, int row, int col);
Vector_t matgetvect(Matrix_t mat, int n);
void matresize(Matrix_t *mat, int n);
void matprint(Matrix_t mat);
Matrix_t mattranspose(Matrix_t A);
void matnormalize(Matrix_t A);
void matfree(Matrix_t mat);

/* ----- Vector data structure ---------------------------- */
double vectget(Vector_t vect, int ind);
void vectset(Vector_t vect, int ind, double val);
void vectfree(Vector_t vect);

/* ----- Input parsing ------------------------------------ */
int parse_int(char* numstr);
double parse_double(char* numstr);
enum Goal parse_goal(char* goalstr);
int get_cmdline_args(int argc, char** argv);
int parse_dpfile();

/* ----- Spectral clustering components ------------------- */
void wam();
void ddg();
void lnorm();
void jacobi();
void spk(Matrix_t eigenvects, Matrix_t cents);

/* ----- Helper functions --------------------------------- */
double l2_norm(Vector_t a, Vector_t b);
Matrix_t spk_getT();
Matrix_t calc_weighted_adjancency_mat(Matrix_t G);
Matrix_t calc_diagonal_degree_mat(Matrix_t W);
Matrix_t calc_normalized_graph_laplacian(Matrix_t W, Matrix_t D);
int eigengap_heuristic(Vector_t eigenvals, Matrix_t eigenvects);
void sort_eigen(Vector_t eigenvals, Matrix_t eigenvects);

/* ----- Jacobi algorithm --------------------------------- */
Matrix_t perform_jacobi(Matrix_t A, Vector_t eigenvalues);
Matrix_t jacobi_calc_rotation_matrix(int n, int i, int j, double c, double s);
void jacobi_transform_matrix(Matrix_t A, int pivi, int pivj, double c, double s);
void jacobi_get_rotation_params(Matrix_t A, int pivi, int pivj, double *c, double *s);
void jacobi_get_pivot(Matrix_t A, int *i, int *j);

/* ----- K-means algorithm -------------------------------- */
void kmeans_setup(Matrix_t eigenvects, Matrix_t cents);
double distance(double* point1, double* point2);
void add_datapoint(double* point1, double* point2);
void div_datapoint(double* point, int d);
void assign_datapoints();
int update_centroids();
void perform_kmeans();
void print_kmeans_centroids();


/* ===== Constants ================================================== */
#define MAX_DATAPOINTS_CNT 50
#define MAX_LINE_LENGTH 2560
#define KMEANS_MAX_ITER 300
#define JACOBI_MAX_ITER 100
#define JACOBI_EPSILON 1.0e-15
#define ERR_INPUT 	"Invalid Input!"
#define ERR_GENERAL "An Error Has Occured"


#endif
