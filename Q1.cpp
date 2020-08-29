/*
 * Alyxandra Spikerman
 * High Perfomance Computing
 * Homework 4 - Question 1
 *
 * Matrix Inversion Using LU Decomposition and OpenMP
 */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <ctime>
#include <omp.h>

const int N = 1024;

using namespace std;

// from matrix-matrix.c
double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC,  &t);
    return (t.tv_sec)+(t.tv_nsec*1e-9);
}

int main() {
    srand(time(NULL)); // seed the random number generator

    // allocate matrices and array p
    int** a = new int*[N];
    int* p = new int[N];
    for(int i = 0; i < N; i++) {
        a[i] = new int[N];
        p[i] = i; // initialize p, the permutation array
    }

    // initialize values in a
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i][j] = (rand() % 50) + 1; // for dense matrices

            // for sparse matrices
            // if (a[i][j] % 2 == 0) {
            //     a[i][j] = 0;
            // }
        }
    }

    int max_index, temp, a_max, *temp_a;

    double start_clock = CLOCK();

    // algorithm for LU decomoposition modified from:
    // - https://www.caam.rice.edu/~zhang/caam335/F09/handouts/lu.pdf
    // - https://www.geeksforgeeks.org/l-u-decomposition-system-linear-equations/
    // - https://en.wikipedia.org/wiki/LU_decomposition
    // - https://eng.umd.edu/~nsw/chbe250/lu-manual.pdf

    // start decompose
    for (int i = 0; i < N; i++) {
        a_max = 0;
        max_index = i;

        // find max index for pivoting
        for (int j = i; j < N; j++) {
            if (a_max < abs(a[j][i])) {
                a_max = abs(a[j][i]);
                max_index = j;
            }
        }

        if (i != max_index) { // pivot a and p
            temp_a = a[i];
            a[i] = a[max_index];
            a[max_index] = temp_a;

            temp = p[i];
            p[i] = p[max_index];
            p[max_index] = temp;
        }

        for (int j = i + 1; j < N; j++) {
            a[j][i] =  a[j][i] / a[i][i];
            for (int m = i + 1; m < N; m++) {
                a[j][m] = a[j][m] - (a[j][i] * a[i][m]);
            }
        }
    }
    // end decompose

    // now pa = lu in lu decomposition and we can invert the matrix

    // initialize the inverse matrix using the permutation array
    int inv_a[N][N];
    for(int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (p[j] != i) {
                inv_a[j][i] = 0;
            } else {
                inv_a[j][i] = 1;
            }
        }
    }

    double start_invert = CLOCK();
    // start invert
    // switch i and j because the resulting matrix is the transpose of the inversed matrix,
    // so have to transpose it here to be in the correct order
    #pragma omp parallel for num_threads(10) shared(inv_a, a)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int m = 0; m < j; m++) {
                inv_a[j][i] = inv_a[j][i] - (inv_a[m][i] * a[j][m]);
            }
        }
        for (int j = N - 1; j >= 0; j--) {
            for (int m = j + 1; m < N; m++) {
                inv_a[j][i] = inv_a[j][i] - (inv_a[m][i] * a[j][m]);
            }
            inv_a[j][i] = inv_a[j][i] / a[j][j];
        }
    }
    // end invert
    double end_clock = CLOCK();
    cout << "Total time (decompose & invert): " << (end_clock - start_clock) << " seconds" << endl;
    cout << "Total parallel time: " << (end_clock - start_invert) << " seconds" << endl;

    // print inverted matrix
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         cout << inv_a[i][j] << " ";
    //     }
    //     cout << endl;
    // }

    // delete allocated matrices
    for(int i = 0; i < N; i++) {
        delete[] a[i];
    }
    delete[] a;
    delete[] p;

    return 0;
}
