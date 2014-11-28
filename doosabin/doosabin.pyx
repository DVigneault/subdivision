# doosabin.pyx

# Imports
import numpy as np
cimport numpy as np
np.import_array()

# Requires common/python on `PYTHONPATH`.
from argcheck import check_ndarray_or_raise, check_type_or_raise
from face_array import (raw_face_array_to_sequence,
                        sequence_to_raw_face_array)

# Type
ctypedef np.float64_t DTYPE_t
DTYPE = np.float64

cdef extern from '<vector>' namespace 'std':
    cdef cppclass vector[T]:
        T& operator[](int)
        int size()

cdef extern from 'doosabin_pyx.h':
    cdef cppclass Surface_cpp 'Surface':
        Surface_cpp(np.ndarray)

        void M(int, double*, double*, double*)
        void Mu(int, double*, double*, double*)
        void Mv(int, double*, double*, double*)
        void Muu(int, double*, double*, double*)
        void Muv(int, double*, double*, double*)
        void Mvv(int, double*, double*, double*)

        int number_of_vertices()
        int number_of_faces()
        int number_of_patches()
        vector[int]& patch_vertex_indices(int)

        object UniformParameterisation(int)

cdef class Surface:
    cdef Surface_cpp* _surface

    def __cinit__(self, T):
        # TODO Check `T`.
        self._surface = new Surface_cpp(sequence_to_raw_face_array(T))

    def __dealloc__(self):
        del self._surface

    def M(self, np.ndarray[np.int32_t, ndim=1, mode='c'] p,
                np.ndarray[DTYPE_t, ndim=2, mode='c'] U,
                np.ndarray[DTYPE_t, ndim=2] X):
        check_ndarray_or_raise('p', p, np.int32, 1, None)
        cdef Py_ssize_t n = p.shape[0]
        check_ndarray_or_raise('U', U, DTYPE, 2, (n, 2))
        check_ndarray_or_raise('X', X, DTYPE, 2,
                               (self._surface.number_of_vertices(), 3))

        cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] R = np.empty(
            (n, 3), dtype=DTYPE)
        cdef np.ndarray[np.int32_t, ndim=1] argsort_p = np.require(
            np.argsort(p), dtype=np.int32)

        cdef Py_ssize_t i, j, k, pj, p0 = -1
        cdef np.ndarray pX

        for i in range(n):
            j = argsort_p[i]
            pj = p[j]
            if pj != p0:
                m = self._surface.patch_vertex_indices(pj).size()
                pX = np.empty((m, 3), dtype=DTYPE)
                for k in range(m):
                    pX[k] = X[self._surface.patch_vertex_indices(pj)[k]]
                p0 = pj

            self._surface.M(p[j], <DTYPE_t*>U.data + 2 * j, <DTYPE_t*>pX.data,
                            <DTYPE_t*>R.data + 3 * j)

        return R

    def uniform_parameterisation(self, np.int32_t N):
        p, U, T = self._surface.UniformParameterisation(N)
        # Return `T` as a list of list objects.
        return p, U, raw_face_array_to_sequence(map(int, T))
