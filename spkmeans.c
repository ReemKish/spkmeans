/* ===== spkmeans.c =================================================
 *
 * 	k - number of required clusters. If equal 0, eigengap heuristic is used.
 */
/* ===== Includes =================================================== */
#include <stdlib.h>
#include <stdio.h>
#include <error.h>
#include <errno.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <float.h>
#include "spkmeans.h"

char *dp_filename = NULL;
/* for kmeans */
double *datapoints, *centroids;
int K, D, N;
Matrix_t dp_mat;
enum Goal goal;

/* ===== Code ======================================================= */
/* ==============================================
 *      Matrix data structure                     
 * ============================================== */
/*
 * Creates a new matrix and initializes all cells to zero.
 */
Matrix_t matcreate(int rows, int cols, enum MatType type) {
	Matrix_t mat;
	mat.rows = rows;
	mat.cols = cols;
	mat.type = type;
	if (NULL == (mat.items = calloc(rows * cols, sizeof(double))))
		error(EXIT_FAILURE, 0, ERR_GENERAL);
	memset(mat.items, 0, rows * cols * sizeof(double));
	return mat; }

/*
 * Sets the value `val` to the cell at row `row` and column `col`.
 */
void matset(Matrix_t mat, int row, int col, double val) {
	if(row < 0 || row >= mat.rows || col < 0 || col >= mat.cols)
		error(EXIT_FAILURE, 0, ERR_GENERAL);
	if(mat.type == ROW_MAJOR)
		mat.items[col + mat.cols * row] = val;
	else
		mat.items[row + mat.rows * col] = val;
}

/*
 * Returns the value of the cell at row `row` and column `col`.
 */
double matget(Matrix_t mat, int row, int col) {
	if(mat.type == ROW_MAJOR)
		return mat.items[col + mat.cols * row];
	else
		return mat.items[row + mat.rows * col];
}

/*
 * Returns a new vector which is the n-th row/column in `mat`.
 * Row/Column depends on wether the matrix is row-major or column-major.
 */
Vector_t matgetvect(Matrix_t mat, int n) {
	Vector_t vect;
	if(mat.type == ROW_MAJOR) {
		vect.size = mat.cols;
		vect.items = &mat.items[n * mat.cols];
	} else {
		vect.size = mat.rows;
		vect.items = &mat.items[n * mat.rows];
	}
	return vect;
}

/*
 * Resizes the matrix.
 * For row-major matrices, changes the number of rows.
 * For column-major matrices, changes the number of columns.
 * Possibly discards cells from the matrix.
 */
void matresize(Matrix_t *mat, int n) {
	if(mat->type == ROW_MAJOR)
		mat->rows = n;
	else
		mat->cols = n;
	if (NULL == (mat->items = realloc(mat->items, mat->rows * mat->cols * sizeof(double))))
		error(EXIT_FAILURE, 0, ERR_GENERAL);
}


Matrix_t mattranspose(Matrix_t A){
	int n = A.rows;
	int i, j;
	Matrix_t At = matcreate(n, n, COL_MAJOR);
    for(i = 0; i < n; i++) {
			for (j = i; j < n; j++) {
				matset(At, i, j, matget(A, j, i));
				matset(At, j, i, matget(A, i, j));
		}
	}
	return At;
}

/*
 * Normalizes the matrix's rows in-place to have unit length.
 */
void matnormalize(Matrix_t A){
	/*normalize the rows*/
	int n = A.rows;
	int k = A.cols;
	int i, j;
	double sum = 0;
	for(i = 0; i < n; i++){
		sum = 0;
		for (j = 0; j < k; j++)
			sum += pow(matget(A, i, j),2);
		sum = sqrt(sum);
		for(j = 0; j < k; j++)
			matset(A, i, j, matget(A, i, j) / sum);
	}
	/*now we need to kmeans the rows*/
	/*for python use the same code except let the start centroids be the ones from the kmeans++*/
}

/*
 * Multiplies two square matrices.
 * The first matrix (A) is modified to be the product of A and B.
 * Returns the modified A.
 */
Matrix_t matmul(Matrix_t A, Matrix_t B) {
	int n = A.rows;
	int i,j,k;
	Matrix_t res = matcreate(n, n, COL_MAJOR);	
	for (i = 0; i < n; i++)
			for (j = 0; j < n; j++)
					for (k = 0; k < n; k++)
							matset(res, i ,j, matget(res, i, j) + matget(A, i, k) * matget(B, k, j));
	for(i = 0; i < n; i++)
		for(j = 0; j <n; j++)
			matset(A, i, j, matget(res, i, j));
	matfree(res);
	return A;
}

/*
 *  Calculates and returns off^2(A) - defied as the sum of 
 *  squares of all off-diagonal elements of A.
 */
double matoffsqr(Matrix_t A) {
	int n = A.rows;
	double res = 0;
	int i, j;
	for (i = 0; i < n; i++){
		for (j = 0; j < n; j++){
			if (i != j)
				res += matget(A, i, j) * matget(A, i, j);
		}
	}
	return res;
}

/*
 * Prints the matrix to stdout.
 */
void matprint(Matrix_t mat) {
	int i,j;
	double val;
	for(i = 0; i < mat.rows; i++) {
		for(j = 0; j < mat.cols; j++) {
			val = matget(mat, i, j);
			if(-5e-5 < val && val < 5e-5) val = 0;
			printf("%.4f", val);
			printf(j == mat.cols - 1 ? "\n" : ",");
		}
	}
}

/*
 * Frees the memory used by the matrix's items.
 */
void matfree(Matrix_t mat) { free(mat.items); }
/* ==============================================
 *      Vector data structure
 * ============================================== */
/*
 * Creates a new vector and initializes all cells to zero.
 */
Vector_t vectcreate(int size) {
	Vector_t vect;
	vect.size = size;
	if (NULL == (vect.items = calloc(size, sizeof(double))))
		error(EXIT_FAILURE, 0, ERR_GENERAL);
	memset(vect.items, 0, size * sizeof(double));
	return vect;
}
/*
 * returns the value of the cell at index `ind`.
 */
double vectget(Vector_t vect, int ind) { return vect.items[ind]; }

/*
 * Sets the value of the cell at index `ind` to `val`.
 */
void vectset(Vector_t vect, int ind, double val) { vect.items[ind] = val; }

/*
 * Sorts the vector in-place using bubble sort.
 */
void vectsort(Vector_t V) {
	int n = V.size;
	double cur1, cur2;
	int i, j;
	for(i = 0; i < n; i++){
		for(j = 0; j < n - 1; j++){
			cur1 = vectget(V, j);
			cur2 = vectget(V, j + 1);
			if (cur1 > cur2){
				vectset(V, j, cur2);
				vectset(V, j + 1, cur1);
			}
		}
	}
}	

/*
 * Prints the vector to stdout.
 */
void vectprint(Vector_t vect) {
	int i;
	double val;
	for(i = 0; i < vect.size; i++) {
			val = vectget(vect, i);
			if(-0.00005 < val && val < 0.00005) val = 0;
			printf("%.4f", val);
			printf(i == vect.size - 1 ? "\n" : ",");
	}
}

/*
 * Frees the memory used by the vector's items.
 */
void vectfree(Vector_t vect) { free(vect.items); }

/* ==============================================
 *      Input parsing
 * ============================================== */
/*
 * Parses the given string into an unsigned number and returns it.
 * In case of error, returns -1.
 */
int parse_int(char* numstr)
{
	char *endptr;
	int num;
	errno = 0;
	num = strtol(numstr, &endptr, 10);
	if(errno == 0 && *numstr != '\0' && (*endptr == '\0' || *endptr == '\n'))
		/* string is valid */
		return num;
	return -1;
}

/*
 * Parses the given string into a double and returns it.
 * In case of error, prints ERR_INPUT and exits the program.
 */
double parse_double(char* numstr)
{
	char *endptr;
	double num;
	errno = 0;
	num = strtod(numstr, &endptr);
	if(errno == 0 && *numstr != '\0' && (*endptr == '\0' || *endptr == '\n'))
		/* string is valid */
		return num;
	error(EXIT_FAILURE, 0, ERR_INPUT);
	return 0;
}

/*
 * Parses the given string as a goal identifier.
 * A valid goal identifier string is one of the following:
 * 	"spk", "wam", "ddg", "lnorm", "jacobi"
 * If the string matches none, returns GOAL_ERR, else the matching enum value.
 */
enum Goal parse_goal(char* goalstr)
{
	if(strcmp("spk", goalstr) == 0)
		return GOAL_SPK;
	else if(strcmp("wam", goalstr) == 0)
		return GOAL_WAM;
	else if(strcmp("ddg", goalstr) == 0)
		return GOAL_DDG;
	else if(strcmp("lnorm", goalstr) == 0)
		return GOAL_LNORM;
	else if(strcmp("jacobi", goalstr) == 0)
		return GOAL_JACOBI;
	return GOAL_ERR;
}

/*
 * Parses the command line argument and updates the corresponding global variables.
 * Cmdline arguments are: k, goal and file_name.
 * Returns 0 on success, -1 on error.
 */
int get_cmdline_args(int argc, char** argv)
{
	/* argument count must be 3 (+1 for module name) */
	if(argc != 4)
		return -1;
	/* parse k, goal */
	if(-1 == (K = parse_int(argv[1])))
		return -1;
	if(GOAL_ERR == (goal = parse_goal(argv[2])))
		return -1;
	dp_filename = argv[3];
	/* no errors */
	return 0;
}

/*
 * Strips the string of both trailing and leading whitespaces.
 * Taken from Stackoverflow:
 * stackoverflow.com/questions/122616/how-do-i-trim-leading-trailing-whitespace-in-a-standard-way
 */
char *stripws(char *str)
{
  char *end;

  /* strip leading space */
  while(isspace((unsigned char)*str)) str++;
  if(*str == 0)  /* All spaces? */
    return str;
  /* strip trailing space */
  end = str + strlen(str) - 1;
  while(end > str && isspace((unsigned char)*end)) end--;
  /* write new null terminator character */
  end[1] = '\0';
  return str;
}

/*
 * Parses the input datapoints file.
 * Returns -1 on error, else 0.
 */
int parse_dpfile()
{
	FILE *datafile = fopen(dp_filename, "r");
	char line[MAX_LINE_LENGTH];
	char *elem;
	double val;
	int i;
	N = 0; D = 1;

	if (NULL == fgets(line, MAX_LINE_LENGTH, datafile))
		return -1;
	/* get datapoint size (D) */
	for (i=0; line[i]; i++) { if(line[i]==',') D++;};

	dp_mat = matcreate(MAX_DATAPOINTS_CNT, D, ROW_MAJOR);

	/* iterate over input lines, each line represents a datapoint */
	do {
		elem = strtok(line, ",");
		for (i=1; i<=D; i++) {
			elem = stripws(elem);
			val = parse_double(elem);
			matset(dp_mat, N, i-1, val);
			/* get next element */
			if ((i!=D) == (NULL == (elem = strtok(NULL, ",")))) {
				/* datapoints are not of uniform size */
				return -1;
			}
		}
		N++;
	} while (NULL != fgets(line, MAX_LINE_LENGTH, datafile));
	fclose(datafile);
	matresize(&dp_mat, N);
	return 0; /* no errors */
}


/* ==============================================
 *      Spectral clustering components
 * ============================================== */
void wam()
{
	Matrix_t W;
	W = calc_weighted_adjancency_mat(dp_mat);
	matfree(dp_mat);
	matprint(W);
	matfree(W);
}

void ddg()
{
	Matrix_t W, D;
	W = calc_weighted_adjancency_mat(dp_mat);
	matfree(dp_mat);
	D = calc_diagonal_degree_mat(W);
	matfree(W);
	matprint(D);
	matfree(D);
}

void lnorm()
{
	Matrix_t W, D, Lnorm;
	W = calc_weighted_adjancency_mat(dp_mat);
	matfree(dp_mat);
	D = calc_diagonal_degree_mat(W);
	Lnorm = calc_normalized_graph_laplacian(W, D);
	matprint(Lnorm);
	matfree(W); matfree(D);
}

void jacobi()
{
	Vector_t eigenvals;
	Matrix_t eigenvects, eigenvectsT;
	eigenvals = vectcreate(dp_mat.rows);
	eigenvects = perform_jacobi(dp_mat, eigenvals);
	matfree(dp_mat);
	vectprint(eigenvals);
	vectfree(eigenvals);
	eigenvectsT = mattranspose(eigenvects);
	matprint(eigenvectsT);
	matfree(eigenvectsT);
	matfree(eigenvects);
}

void spk(Matrix_t eigenvects, Matrix_t cents)
{
	kmeans_setup(eigenvects, cents);
	matfree(eigenvects);
	perform_kmeans();
	print_kmeans_centroids();
	free(datapoints); free(centroids);
}



/* ==============================================
 *      Helper functions
 * ============================================== */
/*
 * Calculates norm of two vectors of the same size.
 */ 
double l2_norm(Vector_t a, Vector_t b)
{
	double res = 0;
	int i = 0;
	int n = a.size;
	for(i=0; i<n; i++)
		res += pow(vectget(a, i) - vectget(b, i), 2);
	return sqrt(res);
}

Matrix_t spk_getT() {
	Matrix_t W, D, Lnorm, eigenvects;
	Vector_t eigenvals;
	W = calc_weighted_adjancency_mat(dp_mat);
	matfree(dp_mat);
	D = calc_diagonal_degree_mat(W);
	Lnorm = calc_normalized_graph_laplacian(W, D);
	eigenvals = vectcreate(Lnorm.rows);
	eigenvects = perform_jacobi(Lnorm, eigenvals);
	matfree(W); matfree(D);
	sort_eigen(eigenvals, eigenvects);
	if(K == 0) K = eigengap_heuristic(eigenvals, eigenvects);
	vectfree(eigenvals);
	matresize(&eigenvects, K);
	matnormalize(eigenvects);
	return eigenvects;
}

/*
 * Returns the sign of x (we define sign(0) = 1).
 */
int sign(double x) { return x >= 0 ? 1 : -1; }

/*
 * Calculates and returns the weighted adjacency matrix of the graph
 * represented by matrix G (G's rows are points in the graph).
 */
Matrix_t calc_weighted_adjancency_mat(Matrix_t G)
{
	int i,j = 0;
	int n = G.rows;
	Matrix_t W = matcreate(n, n, COL_MAJOR);
	for(i = 0; i < n; i++){
		for(j = 0; j < n; j++){
			if(i != j){
				matset(W, i, j, exp(-1 * l2_norm(matgetvect(G, i), matgetvect(G, j)) / 2));
			}
		}
	}
	return W;
}

/*
 * Calculates and returns the diagonal degree matrix of the graph
 * represented by the weighted adjacency matrix W.
 */
Matrix_t calc_diagonal_degree_mat(Matrix_t W)
{
	int n = W.rows;
	double sum = 0;
	int i,j = 0;
	Matrix_t D = matcreate(n, n, COL_MAJOR);
	for(i = 0; i < n; i++) {
		for(j = 0; j < n; j++)
			sum += matget(W, i, j);
		matset(D, i, i, sum);
		sum = 0;
	}
	return D;
}

/*
 * Calculates and returns the normalized graph Laplacian of the graph
 * represented by the weighted adjacency matrix W and diagonal degree matrix W.
 */
Matrix_t calc_normalized_graph_laplacian(Matrix_t W, Matrix_t D)
{
	int n = W.rows;
	int i, j;
	Matrix_t res;
	/* raise all elements of D to the power of -1/2 */
	for(i = 0; i < n; i++)
		matset(D, i, i, 1 / sqrt(matget(D, i, i))); /* -1/2 power */
	res = matmul(D, matmul(W, D));
	/* we want to return id - mul_res */
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			/* multiply all elements by -1 */
			matset(res, i, j, -1 * matget(res, i, j));
			/* add 1 to the diagonal*/
			if (i == j)
				matset(res, i, j, matget(res, i, j) + 1);
		}
	}
	return res;
}	

/*
 * Returns K - the number of clusters wanted for kmeans algorithm, using the eigengap heuristic.
 * Matrix `eigenvects` has the original matrix's eigenvectors as its columns.
 * Vector `eigenvals` has the original matrix's eigenvalues as its elements sorted in ascending order (!).
 */
int eigengap_heuristic(Vector_t eigenvals, Matrix_t eigenvects)
{
	int n = eigenvects.rows;
	int cur_max_index = 0;
	double cur_max = 0;
	int i;
	for(i = 0; i < n / 2; i++){
		if (vectget(eigenvals, i + 1) - vectget(eigenvals, i) > cur_max) {
			cur_max_index = i;
			cur_max = vectget(eigenvals, i + 1) - vectget(eigenvals, i);
		}
	}
	return cur_max_index + 1; /* we start at index 1 and not 0 */
}		

 /*
	* bubble sort eigenvalues and eigenvectors
	*/
void sort_eigen(Vector_t eigenvals, Matrix_t eigenvects) {
	int n = eigenvals.size;
	double cur1, cur2, cur3, cur4;
	int i, j, l;
	for (i = 0; i < n; i++){
		for(j = 0; j < n - 1; j ++){
			cur1 = vectget(eigenvals, j);
			cur2 = vectget(eigenvals, j + 1);
			if (cur1 > cur2){
				vectset(eigenvals, j, cur2);
				vectset(eigenvals, j + 1, cur1);
				/*switch between column j and column j + 1*/
				for (l = 0; l < n; l ++){
					cur3 = matget(eigenvects, l, j);
					cur4 = matget(eigenvects, l, j + 1);
					matset(eigenvects, l, j, cur4);
					matset(eigenvects, l, j + 1, cur3);
				}
			}
		}
	}
}


/* ==============================================
 *      Jacobi algorithm
 * ============================================== */
/*
 * Performs the Jacobi algorithm on matrix A in order to find eigenvalues and eigenvectors.
 * The returned matrix contains A's eigenvectors as its rows.
 * The eigenvalues vector is populated with A's eigenvalues.
 */
Matrix_t perform_jacobi(Matrix_t A, Vector_t eigenvalues)
{
	int n = A.rows;
	int iterations = 0;
	int i;
	int pivi, pivj;  /* coordinates of the rotation pivot */
	double c, s;  /* rotation matrix parameters */
	double offsqr;  /* = off^2(A) (see matoffsqr() documentation)*/
	Matrix_t P, V = matcreate(n, n, COL_MAJOR);
	for(i=0; i<n; i++) matset(V, i, i, 1);  /* set V to the unity matrix */
	while (iterations++ < JACOBI_MAX_ITER) {
		/*printf("==============================================================\n", iterations);
		printf("                   Jacobi iteration %d\n", iterations);
		printf("==============================================================\n", iterations);
		printf("--[%d]-- ---------- Matrix A ----------\n", iterations);
		matprint(A);
		printf("\n");*/
		jacobi_get_pivot(A, &pivi, &pivj);
		/*printf("--[%d]-- Pivot = A[%d,%d] = %f\n", iterations, pivi, pivj, matget(A, pivi, pivj));*/
		jacobi_get_rotation_params(A, pivi, pivj, &c, &s);
		/*printf("--[%d]-- : c = %f, s = %f\n", iterations, c, s);*/
		P = jacobi_calc_rotation_matrix(n, pivi, pivj, c, s);
		offsqr = matoffsqr(A);
		jacobi_transform_matrix(A, pivi, pivj, c, s);
		/*printf("--[%d]-- ---------- Matrix A' (1) (Via matrix multiplication) ----------\n", iterations);
		matprint(Atag);
		printf("\n");
		printf("--[%d]-- ---------- Matrix A' (2) (Via shortcut) ----------\n", iterations);
		matprint(A);
		printf("\n");*/
		/*printf("--[%d]-- : off^2(A) = %f\n", iterations, offsqr);
		printf("--[%d]-- : off^2(A') = %f (via shortcut)\n ()", iterations, temp);*/
		matmul(V, P); /* V is the product of all P */
		matfree(P);
		/*printf("--[%d]-- ---------- Matrix V ----------\n", iterations);
		matprint(V);
		printf("\n\n");*/
		if(offsqr - matoffsqr(A) <= JACOBI_EPSILON) /* check convergence */
			break;
	}
	for(i = 0; i < n ; i++)
		vectset(eigenvalues, i, matget(A, i, i)); /* diagonal of A are the eigenvalues */
	return V;
}


/*
 * Calculates and returns the rotation matrix P of the jacobi algorithm.
 * 
 */
Matrix_t jacobi_calc_rotation_matrix(
		int n,  /* matrix size */
		int pivi, int pivj,  /* rotation pivot coordinates */
		double c, double s /* rotation parameters */)
{
	int k;
	Matrix_t P = matcreate(n, n, COL_MAJOR);
  for (k = 0; k < n; k++) /* sets diagonal to 1 */
		matset(P, k, k, 1);
	matset(P, pivi, pivi, c);
	matset(P, pivj, pivj, c);
	matset(P, pivi, pivj, s);
	matset(P, pivj, pivi, -1 * s);
	return P;
}


/*
 *  Get the jacobi rotation matrix parameters: c and s.
 *  pivi & pivj are the coordinates of Aij (the rotation pivot) in A.
 */
void jacobi_get_rotation_params(Matrix_t A, int pivi, int pivj, double *c, double *s)
{
	double theta, t;
	double Ajj = matget(A, pivj, pivj);
	double Aii = matget(A, pivi, pivi);
	double Aij = matget(A, pivi, pivj);
	theta = ((Ajj - Aii) / (2 * Aij));
	t = sign(theta) / (fabs(theta) + sqrt(theta * theta + 1));
	*c = 1 / sqrt(t * t + 1);
	*s = (*c) * t;
}


/*
 * Calculates the pivot for the Jacobi algorithm.
 * The pivot is the off-diagonal element in A with the largest absolute value.
 * Sets i,j to be the row and column of the pivot.
 */
void jacobi_get_pivot(Matrix_t A, int *i, int *j)
{
	int n = A.rows;
	int row, col;
	*i = 0;
	*j = 1;
	for(row = 0; row < n; row++) {
		for(col = 0; col < n; col++) {
			if (row != col /* element must be off-diagonal */ &&
					fabs(matget(A, *i, *j)) < fabs(matget(A, row, col))) {
				*i = row;
				*j = col;
			}
		}
	}
}


/*
 * Transform matrix A to A' as described in the Jacobi algorithm.
 */
void jacobi_transform_matrix(
		Matrix_t A,
		int pivi, int pivj,  /* rotation pivot coordinates */
		double c, double s /* rotation parameters */)
{
	int n = A.rows;
	double Ari, Arj, Aii, Ajj;
	int r;
	for (r = 0; r < n; r++){
		if (r != pivi && r != pivj) {
			Ari = matget(A, r, pivi);
			Arj = matget(A, r, pivj);
			matset(A, r, pivi, c * Ari - s * Arj);
			matset(A, pivi, r, c * Ari - s * Arj);
			matset(A, r, pivj, c * Arj + s * Ari);
			matset(A, pivj, r, c * Arj + s * Ari); }
	}
	Aii = matget(A, pivi, pivi);
	Ajj = matget(A, pivj ,pivj);
	matset(A, pivi, pivi, c * c * Aii + s * s * Ajj - 2 * s * c * matget(A, pivi, pivj));
	matset(A, pivj, pivj, s * s * Aii + c * c * Ajj + 2 * s * c * matget(A, pivi, pivj));
	matset(A, pivi, pivj, 0);
	matset(A, pivj, pivi, 0);
}


/* ==============================================
 *      K-means algorithm
 * ============================================== */

/*
 * Setup the required variables and datatype conversions before invoking kmeans.
 */
void kmeans_setup(Matrix_t eigenvects, Matrix_t cents) 
{
	int i, j;
	if (NULL == (datapoints = (double*) calloc(N, (1+K)*sizeof(double))))
		error(EXIT_FAILURE, 0, ERR_GENERAL);
	if (NULL == (centroids = (double*) calloc(K, K*sizeof(double))))
		error(EXIT_FAILURE, 0, ERR_GENERAL);
	for(i = 0; i < N; i++) {
		for(j = 0; j < K; j++)
			datapoints[j + i * (K+1) + 1] = matget(eigenvects, i, j);
		datapoints[i * (K+1)] = 0;
		if(i < K && cents.type == MAT_INVALID)
			memcpy(&centroids[i*K], &datapoints[i*(K+1)+1], K*sizeof(double));
	}
	if(cents.type != MAT_INVALID)
		for(i = 0; i < K; i++)
			memcpy(&centroids[i*K], matgetvect(cents, i).items, K*sizeof(double));
	/*printf("centroids[%d]=%f\n", 0, centroids[0]);
	printf("centroids[%d]=%f\n", 1, centroids[1]);
	printf("centroids[%d]=%f\n", 2, centroids[2]);
	printf("centroids[%d]=%f\n", 3, centroids[3]);
	printf("centroids[%d]=%f\n", 4, centroids[4]);
	printf("centroids[%d]=%f\n", 5, centroids[5]);
	need to fix datapoints*/
}


void perform_kmeans()
{
	int i;
	for (i = 0; i < KMEANS_MAX_ITER; i++) {
		assign_datapoints();
		if (update_centroids())  /* reached convergence */
			break;
	}
}


double distance(double* point1, double* point2)
{
	double dist = 0;
	int i;
	for (i=0; i < K; i++)
		dist += (point1[i] - point2[i]) * (point1[i] - point2[i]);
	return dist;
}


void add_datapoint(double* point1, double* point2)
/* Adds point1 to point2 and stores the result in point1 */
{
	int i;
	for(i=0; i<K; i++)
		point1[i] = point1[i] + point2[i];
}


void div_datapoint(double* point, int d)
/* divides point by interger d and stores the result in point*/
{
	int i;
	for(i=0; i<K; i++)
		point[i] = point[i] / d;
}


void assign_datapoints()
{
	int i, j, min_i;
	double dist, min_dist;
	for (j = 0; j < N; j++) {
		min_i = 0;
		min_dist = DBL_MAX;
		for (i=0; i<K; i++) {
			dist = distance(&centroids[i*K],&datapoints[j*(K+1)+1]);
			min_i = dist < min_dist ? i : min_i;
			min_dist = dist < min_dist ? dist : min_dist;
		}
		datapoints[j*(K+1)] = (double) min_i;
	}
}


int update_centroids()
/* returns 1 if reached convergence, else 0 */
{
	int converges;
	int i, j;
	int *centroid_sizes = calloc(K, sizeof(int));
	double *old_centroids = calloc(K*K, sizeof(double));
	
	memcpy(old_centroids, centroids, K*K*sizeof(double));
	/* zero the centroids & sizes array */
	for(i=0; i<K; i++) {
		centroid_sizes[i] = 0;
		for (j=0; j<K; j++)
			centroids[i*K+j] = 0;
	}

	/* iterate over datapoints and update sum */
	for (i=0; i < N; i++) {
		j = datapoints[i*(K+1)];
		add_datapoint(&centroids[j*K], &datapoints[i*(K+1)+1]);
		centroid_sizes[j]++;
	}
	
	/* divide each centroid by the cluster size to obtain the mean */
	for(i=0; i<K; i++)
		div_datapoint(&centroids[i*K], centroid_sizes[i]);

	/* check convergence by comparing to previous centroids */
	converges = 1;
	for(i=0; i<K*K; i++)
		if (centroids[i] != old_centroids[i])
			converges = 0;

	free(centroid_sizes);
	free(old_centroids);
	return converges;
}


void print_kmeans_centroids()
{
	int i,j;
	double val;
	for(i=0; i<K; i++) {
		val = centroids[i*K];
		if(-5e-5 < val && val < 5e-5) val = 0;
		printf("%.04f", val);
		for(j=1; j<K; j++) {
			val = centroids[i*K+j];
			if(-5e-5 < val && val < 5e-5) val = 0;
			printf(",%.04f", val);
		}
		printf("\n");
	}
}


int main(int argc, char **argv)
{
	Matrix_t cents;
	cents.type = MAT_INVALID;
	if(-1 == get_cmdline_args(argc, argv))
		error(EXIT_FAILURE, 0, ERR_INPUT);
	if(-1 == parse_dpfile())
		error(EXIT_FAILURE, 0, ERR_INPUT);
	switch(goal) {
		case GOAL_WAM:
			wam();
			break;
		case GOAL_DDG:
			ddg();
			break;
		case GOAL_LNORM:
			lnorm();
			break;
		case GOAL_JACOBI:
			jacobi();
			break;
		case GOAL_SPK:
			spk(spk_getT(), cents);
			break;
		default:
			break;
	}
	return EXIT_SUCCESS;
}
