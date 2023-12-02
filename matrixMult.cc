#include <cstdint>
#include <iostream>
#include <cstdlib>
#include <string>
#include <cstring>
#include <chrono>
#include <cassert>
#include <thread>
#include <immintrin.h>

using std::cout;
using std::cerr;
using std::endl;
using std::stoi;
using std::thread;
using namespace std::chrono;

// single-threaded matrix multiply
void singleMatMult( int32_t m, int32_t ** operand1, int32_t ** operand2, int32_t ** ouput ) {
    for ( int32_t i = 0; i < m; i++ ) {
        for ( int32_t j = 0; j < m; j++ ) {
            for ( int32_t k = 0; k < m; k++ ) {
                ouput[i][j] += operand1[i][k] * operand2[i][k];
            }
        }
    }
}

// multi-threaded matrix multiply
void thdMatMultMain( int32_t m, int32_t ** operand1, int32_t ** operand2, int32_t ** ouput, int32_t start, int32_t end ) { // C_TODO: try mixing both SIMD and thds
    for ( int32_t i = start; i < end; i++ ) {
        for ( int32_t j = 0; j < m; j++ ) {
            for ( int32_t k = 0; k < m; k++ ) {
                ouput[i][j] += operand1[i][k] * operand2[i][k];
            }
        }
    }
}

void thdMatMult( int32_t m, int32_t ** operand1, int32_t ** operand2, int32_t ** ouput, int32_t nThreads ) {
    int32_t sliceSize = m/nThreads, currSlice = 0;
    thread ** threads = new thread*[nThreads];
    for ( int32_t i = 0; i < nThreads; i++ ) {
        threads[i] = new thread( thdMatMultMain, m, operand1, operand2, ouput, currSlice, currSlice + sliceSize );
        currSlice += sliceSize;
    }

    for ( int32_t i = 0; i < nThreads; i++ ) {
        threads[i]->join();
        delete threads[i];
    }

    delete threads;
}

// SIMD/AVX matrix multiply
void simdMatMult( int32_t m, int32_t ** operand1, int32_t ** operand2, int32_t ** ouput ) {
    __m256i accumulator, operandVec1, operandVec2;
    __m128i extract;
    for ( int32_t i = 0; i < m; i++ ) {
        for ( int32_t j = 0; j < m; j++ ) {
            accumulator = _mm256_set_epi32( 0, 0, 0, 0, 0, 0, 0, 0 );
            for ( int32_t k = 0; k < m; k += 8 ) {
                operandVec1 = _mm256_load_si256( (__m256i const*)&operand1[i][k] );
                operandVec2 = _mm256_load_si256( (__m256i const*)&operand2[i][k] );

                // reuse operand 1 for output
                operandVec1 = _mm256_mullo_epi32( operandVec1, operandVec2 );
                accumulator = _mm256_add_epi32( operandVec1, accumulator );
            }
            operandVec1 = _mm256_permute2f128_si256( accumulator, accumulator, 1 ); // C_TODO: try using normal adds instead of hadds for condensing
            operandVec2 = _mm256_hadd_epi32( accumulator, operandVec1 );
            operandVec1 = _mm256_permute4x64_epi64( operandVec2, 14 );
            accumulator = _mm256_hadd_epi32( operandVec2, operandVec1 );
            operandVec1 = _mm256_hadd_epi32( accumulator, accumulator );
            extract = _mm256_extracti128_si256( operandVec1, 0 );
            ouput[i][j] = (int32_t)_mm_extract_epi32(extract, 0);
        }
    }
}


// CUDA matrix multiply
void cudaMatMult( int32_t m, int32_t ** operand1, int32_t ** operand2, int32_t ** ouput ) {
    // C_TODO learn CUDA and implement this routine
}

// returns a M x M matrix filled with random values
int32_t ** generateMatrix( int32_t m ) {
    int32_t ** retval = new int32_t *[m];
    for ( int32_t i = 0; i < m; i++ ) {
        retval[i] = static_cast<int32_t *>(aligned_alloc( 32 * 8 , sizeof(int32_t) * m )); // needs to be 32-byte aligned for SIMD
        for ( int32_t j = 0; j < m; j++ ) {
            retval[i][j] = rand();
        }
    }
    return retval;
}

// returns an zero-filled M x M matrix
int32_t ** generateEmptyMatrix( int32_t m ) {
    alignas(16) int32_t ** retval = new int32_t *[m];
    for ( int32_t i = 0; i < m; i++ ) {
        retval[i] = static_cast<int32_t *>(aligned_alloc( 32 * 8 , sizeof(int32_t) * m )); // needs to be 32-byte aligned for SIMD
        memset( retval[i], 0, m * sizeof(retval[i][0]) );
    }
    return retval;
}

// returns an zero-filled M x M matrix
void emptyMatrix( int32_t m,  int32_t ** mat ) {
    for ( int32_t i = 0; i < m; i++ ) {
        memset( mat[i], 0, m * sizeof(mat[i][0]) );
    }
}

void deleteMatrix( int32_t m, int32_t ** mat ) {
    for ( int32_t i = 0; i < m; i++ ) {
        free( mat[i] );
        // delete mat[i];
    }
    delete mat;
}

void checkMatrices( int32_t m, int32_t ** matA, int32_t ** matB ) {
    for ( int32_t i = 0; i < m; i++ ) {
        for ( int32_t j = 0; j < m; j++ ) {
            if ( matA[i][j] != matB[i][j] ) {
                cout << "MISMATCH: at" << endl;
                cout << "i: " << i << ", j: " << j << ", matA val: " << matA[i][j] << ", matB val: " << matB[i][j] << endl;
            }
            assert( matA[i][j] == matB[i][j] );
        }
    }
}

void checkResults( int32_t m, int32_t ** matA, int32_t ** matB ) {
    // check results against eachother
    checkMatrices( m, matA, matB );
    emptyMatrix( m, matB ); // clear for reuse
}

void printUsage( char * argv[] ) {
    cerr << "Usage: " << argv[0] << " [ matrix_size (> 0) ] [ nThreads (> 0) ]" << endl;
    exit( EXIT_FAILURE );
}


int main( int argc, char * argv[] ) {
    int32_t m = 512, nThreads = 1;
    switch ( argc ) {
        case 3:
			nThreads = stoi( argv[2] );
			if ( m < 1 ) printUsage( argv );
	    case 2:
			m = stoi( argv[1] );
			if ( m < 1 ) printUsage( argv );
	  case 1:											// use defaults
		break;
	  default:
		printUsage( argv );
	} // switch

    // type used for simd
    // __m256i test;

    // must have equal work among threads or else it is not a fair comparison
    assert( m % nThreads == 0 );
    // must be able to divide work among simd instructions for fair comparison
    assert( m % 8 == 0 );

    int32_t ** multOperand1 = generateMatrix( m );
    // we treat this matrix as an array of column arrays to allow for faster SIMD/CUDA access
    // all other matrices are treated in the traditional way as an array of row arrays
    int32_t ** multOperand2 = generateMatrix( m );
    int32_t ** outputMat1 = generateEmptyMatrix( m );
    int32_t ** outputMat2 = generateEmptyMatrix( m );

    // start time for timing tests
    time_point<steady_clock> starttime;

    // single-threaded matrix multiply
    cout << "Single: ";
    starttime = steady_clock::now();
    singleMatMult( m, multOperand1, multOperand2, outputMat1 );
	cout << (steady_clock::now() - starttime).count() / 1'000'000'000.0 << endl;

    // multi-threaded matrix multiply
    cout << "Multi: ";
    starttime = steady_clock::now();
    thdMatMult( m, multOperand1, multOperand2, outputMat2, nThreads );
	cout << (steady_clock::now() - starttime).count() / 1'000'000'000.0 << endl;

    // check results against eachother and clear Mat2 for reuse
    checkResults( m, outputMat1, outputMat2 );

    // SIMD/AVX matrix multiply
    cout << "SIMD: ";
    starttime = steady_clock::now();
    simdMatMult( m, multOperand1, multOperand2, outputMat2 );
	cout << (steady_clock::now() - starttime).count() / 1'000'000'000.0 << endl;

    // check results against eachother and clear Mat2 for reuse
    checkResults( m, outputMat1, outputMat2 );

    // CUDA matrix multiply
    cout << "CUDA: ";
    starttime = steady_clock::now();


	cout << (steady_clock::now() - starttime).count() / 1'000'000'000.0 << endl;

    deleteMatrix( m, multOperand1 );
    deleteMatrix( m, multOperand2 );
    deleteMatrix( m, outputMat1 );
    deleteMatrix( m, outputMat2 );

    return 0;
}