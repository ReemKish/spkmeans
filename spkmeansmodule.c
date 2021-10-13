#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "spkmeans.h"

extern int K, D, N;
extern Matrix_t dp_mat;
/* ---------------------------- CPython API ----------------------------- */
static double* pylist2arr(PyObject *seq);
static Matrix_t pylist2matrix(PyObject *seq);
static PyObject* matrix2pylist(Matrix_t A);

/* CPython wrapper function */
static PyObject* parse_py(PyObject *self, PyObject *args)
{
	PyObject *datapoints_seq;
	if (!PyArg_ParseTuple(args, "Oiii", &datapoints_seq, &K, &N, &D))
		return NULL;
	if((dp_mat = pylist2matrix(datapoints_seq)).type == MAT_INVALID)
		return NULL;
	Py_RETURN_NONE;
}

static PyObject* wam_py(PyObject *self, PyObject *args)
{
	if(NULL == parse_py(self, args)) return NULL;
	wam();
	Py_RETURN_NONE;
}

static PyObject* ddg_py(PyObject *self, PyObject *args)
{
	if(NULL == parse_py(self, args)) return NULL;
	ddg();
	Py_RETURN_NONE;
}

static PyObject* lnorm_py(PyObject *self, PyObject *args)
{
	if(NULL == parse_py(self, args)) return NULL;
	lnorm();
	Py_RETURN_NONE;
}

static PyObject* jacobi_py(PyObject *self, PyObject *args)
{
	if(NULL == parse_py(self, args)) return NULL;
	jacobi();
	Py_RETURN_NONE;
}

static PyObject* spk_getT_py(PyObject *self, PyObject *args)
{
	if(NULL == parse_py(self, args)) return NULL;
	Matrix_t T = spk_getT();
	return matrix2pylist(T);
}

static PyObject* spk_py(PyObject *self, PyObject *args)
{
	PyObject *eigenvects_seq, *centroids_seq;
	Matrix_t Tmat, cents;
	if (!PyArg_ParseTuple(args, "OOiii", &eigenvects_seq, &centroids_seq, &K, &N, &D))
		return NULL;
	if( (Tmat = pylist2matrix(eigenvects_seq)).type == MAT_INVALID ||
			(cents = pylist2matrix(centroids_seq)).type == MAT_INVALID)
		return NULL;
	spk(Tmat, cents);
	Py_RETURN_NONE;
}



static double* pylist2arr(PyObject *seq)
/* written with inspiration from https://www.oreilly.com/library/view/python-cookbook/0596001673/ch16s03.html*/
{
	double *arr;
	Py_ssize_t size;
	Py_ssize_t i;
	PyObject *item, *fitem;
	if (!(seq = PySequence_Fast(seq, "argument must be iterable")))
		return NULL;
	size = PySequence_Fast_GET_SIZE(seq);
	if (NULL == (arr = (double*) malloc(size*sizeof(double)))) {
		Py_DECREF(seq);
		return (void*) PyErr_NoMemory();
	}
	for (i=0; i < (Py_ssize_t) size; i++) {
		item = PySequence_Fast_GET_ITEM(seq, i);
		if (NULL == (fitem = PyNumber_Float(item))) {
			Py_DECREF(seq);
			PyErr_SetString(PyExc_TypeError, "all list items must be numbers");
			return NULL;
		}
		arr[i] = PyFloat_AsDouble(fitem);
	}
	return arr;
}

/*
 * Converts a Matrix_t type to python list of lists.
 * The resulted list of lists matrix is row-major (i.e matlst[0] is row 0 of A).
 */
static PyObject* matrix2pylist(Matrix_t A)
{
	PyObject *matlst, *rowlst;
	Py_ssize_t i, j;
	if (NULL == (matlst = PyList_New(A.rows)))
		return PyErr_NoMemory();
	for (i = 0; i < (Py_ssize_t)A.rows; i++) {
		if (NULL == (rowlst = PyList_New(A.cols)))
			return PyErr_NoMemory();
		for (j = 0; j < (Py_ssize_t)A.cols; j++)
			PyList_SetItem(rowlst, j, Py_BuildValue("f", matget(A, i, j)));
		PyList_SetItem(matlst, i, rowlst);
	}
	return matlst;
}

static Matrix_t pylist2matrix(PyObject *seq)
{
	Matrix_t mat;
	Py_ssize_t i, j;
	Py_ssize_t rows, cols;
	double *farr_item;
	PyObject *item;
	if (!(seq = PySequence_Fast(seq, "argument must be iterable")))
		mat.type = -1 ;
	else {
		rows = PySequence_Length(seq);
		cols = PySequence_Length(PySequence_Fast_GET_ITEM(seq, 0));
		mat = matcreate(rows, cols, ROW_MAJOR);
		for (i=0; i < (Py_ssize_t) rows; i++) {
			item = PySequence_Fast_GET_ITEM(seq, i);
			if (NULL == (farr_item = pylist2arr(item))) {
				Py_DECREF(seq);
				mat.type = - 1;
				return mat;
			}
			for (j=0; j < cols; j++) {
				matset(mat, i, j, farr_item[j]);
			}
		}
	}
	return mat;
}

/* Methods definition */
static PyMethodDef spkmeansmoduleMethods[] = {
	{	"wam",            						/* extension method name */
		(PyCFunction) wam_py,					/* the function that implements this method */
		METH_VARARGS,									/* indicates the method receives arguments */
		PyDoc_STR("Prints the weighted adjacency matrix.")	/* Method docstring */
	},
	{	"ddg",            						/* extension method name */
		(PyCFunction) ddg_py,					/* the function that implements this method */
		METH_VARARGS,									/* indicates the method receives arguments */
		PyDoc_STR("Prints the diagonal degree matrix.")	/* Method docstring */
	},
	{	"lnorm",            						/* extension method name */
		(PyCFunction) lnorm_py,					/* the function that implements this method */
		METH_VARARGS,									/* indicates the method receives arguments */
		PyDoc_STR("Prints the normalized graph Laplacian matrix.")	/* Method docstring */
	},
	{	"jacobi",            						/* extension method name */
		(PyCFunction) jacobi_py,					/* the function that implements this method */
		METH_VARARGS,									/* indicates the method receives arguments */
		PyDoc_STR("Prints the eigenvalues and eigenvectors using the jacobi algorithm.")	/* Method docstring */
	},
	{	"spk",            						/* extension method name */
		(PyCFunction) spk_py,					/* the function that implements this method */
		METH_VARARGS,									/* indicates the method receives arguments */
		PyDoc_STR("Prints the final clusters obtained from the spectral clustering algorithm.")	/* Method docstring */
	},
	{	"spk_getT",        						/* extension method name */
		(PyCFunction) spk_getT_py,		/* the function that implements this method */
		METH_VARARGS,									/* indicates the method receives arguments */
		PyDoc_STR("Returns the normalized eigenvectors matrix T component of the spectral clustering algorithm.")	/* Method docstring */
	},
	{NULL, NULL, 0, NULL}           /* acts as a sentinel */
};

/* Module definition */
static struct PyModuleDef moduledef = {
	PyModuleDef_HEAD_INIT,
	"spkmeansmodule",	 			/* name of the module */
	NULL,										/* should be docstring of the module but can be NULL */
	-1,											/* size of per-interpreter state of the module, but we don't keep a state */
	spkmeansmoduleMethods   /* methods provided by the module */
};

/* Module creation */
PyMODINIT_FUNC
PyInit_spkmeansmodule(void)
{
	
	PyObject *module;
	module = PyModule_Create(&moduledef);
	if (!module)
		return NULL;
	return module;
}


