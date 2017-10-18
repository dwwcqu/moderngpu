/******************************************************************************
 * Copyright (c) 2013, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 B*       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 *
 * Code and text by Sean Baxter, NVIDIA Research
 * See http://nvlabs.github.io/moderngpu for repository and documentation.
 *
 ******************************************************************************/

#pragma once

#include "../mgpuhost.cuh"
#include "../kernels/segreduce.cuh"
#include "../kernels/bulkinsert.cuh"

#include "../constants.h"

namespace mgpu {

template<size_t Size, bool LoadLeft>
struct SpmvTuningNormal {
	enum { Indirect = false };
	typedef LaunchBox<
		SegReduceTuning<128, 11, 0, false, false>,
		SegReduceTuning<128, 11, 0, true, false>,
		SegReduceTuning<128, 7, 0, true, false>
	> Tuning;
};

template<size_t Size, bool LoadLeft>
struct SpmmTuningNormal {
	enum { Indirect = false };
	typedef LaunchBox<
		SegReduceTuning<128, 11, 0, false, false>,
		SegReduceTuning<128, 11, 0, true, false>,
		SegReduceTuning<MGPU_NT, MGPU_VT, 0, true, false>
	> Tuning;
};


////////////////////////////////////////////////////////////////////////////////
// CTASpmvLoad
// Loads matrix values and column indices and gathers vector values. Finds 
// products and transposes terms into register output in thread order.

template<int NT, int VT, bool LoadLeft, bool HalfCapacity, typename T,
	typename MulOp>
struct CTASpmmLoad {
	enum {
		NV = NT * VT,
		Capacity = HalfCapacity ? (NV / 2) : NV
	};

  // Note: count2 = how many nnz we need to load in terms of tid
	template<typename MatrixIt, typename ColumnsIt, typename VecIt>
	MGPU_DEVICE static void LoadDirectSpmm(int count2, int tid,
		MatrixIt matrixData[VT], ColumnsIt columns[VT], VecIt vec_global, 
    const int slab, T identity, MulOp mulOp, T data[MGPU_TB] ) {

		// Use ldg to load vector data in strided order.
    #pragma unroll
    for( int ii=0; ii<MGPU_TB; ii++ )
    {
      int col_all = __shfl(columns[   0], ii+slab);
      T   val_all = __shfl(matrixData[0], ii+slab);
    	data[ii]    = val_all*__ldg(vec_global+col_all);
    }

		// Clear out the out-of-range inputs. 
		if( count2 < NV && tid >= count2)
      #pragma unroll
      for( int ii=0; ii<MGPU_TB; ii++ )
        data[ii] = identity;

		// Transpose from strided to thread order.
		/*if(HalfCapacity)
			HalfSmemTranspose<NT, VT*MGPU_TB>(stridedData, tid, storage.data, data);
		else {
      // Cannot unroll, because using smem resource sequentially
      for( int j=0; j<MGPU_TB; j++ )
      {
			  DeviceRegToShared<NT2, VT>(stridedData+j*VT, tid, storage.data+NV*tiy);
			  DeviceSharedToThread<VT>(storage.data+NV*tiy, tid, data+j*VT);
      }
		}*/
	}
};

template<typename Tuning, bool Indirect, bool LoadLeft, typename MatrixIt, 
	typename ColsIt, typename CsrIt, typename SourcesIt, typename VecIt, 
	typename DestIt, typename T, typename MulOp, typename AddOp>
MGPU_LAUNCH_BOUNDS void KernelSpmmCsr(MatrixIt matrix_global,
	ColsIt cols_global, int nz, CsrIt csr_global, SourcesIt sources_global, 
	VecIt vec_global, CsrIt limits_global, DestIt dest_global,
	T* carryOut_global, T identity, MulOp mulOp, AddOp addOp, const int B_ncols) {

	typedef MGPU_LAUNCH_PARAMS Params;
	const int NT = Params::NT;
	const int VT = Params::VT;
	const int NV = NT * VT;
	const bool HalfCapacity = (sizeof(T) > sizeof(int)) && Params::HalfCapacity;

	typedef CTAReduce<NT, AddOp> FastReduce;
	typedef CTASegReduce<NT, VT, HalfCapacity, T, AddOp> SegReduce;
  typedef CTASpmmLoad<NT, VT, LoadLeft, HalfCapacity, T, MulOp> SpmmLoad;

	union Shared {
		int csr[NV + 1];
		typename SegReduce::Storage segReduceStorage;
	};
	__shared__ Shared shared;

	int tid = threadIdx.x;
	int block = blockIdx.x;
	int gid = NV * block;
	int count2 = min(NV, nz - gid);
  int lane_id = tid & (32 - 1);

	// Retrieve the left and right row limits.
	int limit0 = __ldg(limits_global+block);
  int limit1 = __ldg(limits_global+block + 1);

	SegReduceRange range;
	SegReduceTerms terms;
	int rows[MGPU_TB + 1], rowStarts[MGPU_TB];
	T data[MGPU_TB];

  // Transform the row limits into ranges.
  range = DeviceShiftRange(limit0, limit1);
  int numRows = range.end - range.begin;

  //if( tid==0 ) printf("%d,%d,%d,%d\n", block, limit0, limit1, numRows);

  // Load the CSR interval.
  DeviceGlobalToSharedLoop<NT, VT>(numRows, csr_global + range.begin, tid, 
      shared.csr);
  __syncthreads();

  // Load column indices directly from cols_global.
  // Load values into stridedData.
  int columns[VT];
  T matrixData[VT];
	if( tid < count2 )
  {
    columns[0]    = __ldg(cols_global+tid)<<6;
    matrixData[0] = __ldg(matrix_global+tid);
  }
  else
  {
    columns[0]    = 0;
    matrixData[0] = 0.f;
  }

  for( int slab=0; slab<32; slab+=MGPU_TB )
  {
    // Removed Indirect load case
    // This is a direct load so we don't have a data-dependency on the
    // limits.
    SpmmLoad::LoadDirectSpmm(count2, tid,
        matrixData, columns, vec_global+slab+(blockIdx.z<<5), 
        slab, identity, mulOp, data);

    // Flatten CSR->COO and return the segmented scan terms.
    if( lane_id==0 )
      terms = DeviceSegReducePrepare<NT, MGPU_TB>(shared.csr, numRows, tid+slab,
          gid, range.flushLast, rows, rowStarts);
    #pragma unroll
    for( int i=0; i<MGPU_TB+1; i++ )
      rows[i] = __shfl(rows[i], 0);
    #pragma unroll
    for( int i=0; i<MGPU_TB; i++ )
      rowStarts[i] = __shfl(rowStarts[i], 0);
    terms.tidDelta = __shfl(terms.tidDelta, 0);

    // Reduce tile data and store to dest_global. Write tile's carry-out
    // term to carryOut_global.
    SegReduce::ReduceToGlobalSpmm(rows, range.total, terms.tidDelta, 
        range.begin, block, tid, data, dest_global+slab+(blockIdx.z<<5), 
        carryOut_global+slab*gridDim.x,
        identity, addOp, shared.segReduceStorage);
  }
}

template<typename DestIt>
void printDense(const int nrows, const int ncols, DestIt array)
{
  int row_length=std::min(20,nrows);
  //int row_length=std::min(20,nrows);
  int col_length=std::min(20,ncols);

  std::cout << row_length << " " << col_length << std::endl;

  // Allocate array on host
  int nvals=nrows*ncols;
  float* temp = (float*) malloc(nvals*sizeof(float));
  CUDA( cudaMemcpy( temp, array, nvals*sizeof(float), 
      cudaMemcpyDeviceToHost ));

  // Print out all dense values
  std::cout << "dest_global:\n";
  for( int i=0;i<min(40,nvals);i++ )
    std::cout << "[" << i << "]:" << temp[i] << " ";
  std::cout << "\n";

  // Print in matrix format
  for( int row=0; row<row_length; row++ ) {
    for( int col=0; col<col_length; col++ ) {
      // Print row major order matrix in row major order
      if( temp[row*ncols+col]!=0.0 ) std::cout << "x ";
      else std::cout << "0 ";
    }
    std::cout << std::endl;
  }

  // Cleanup
  if( temp ) free( temp );
}

template<typename Tuning, bool Indirect, bool LoadLeft, typename MatrixIt, 
	typename ColsIt, typename CsrIt, typename SourcesIt, typename VecIt, 
  typename LimIt, typename DestIt, typename T, typename MulOp, typename AddOp>
MGPU_HOST void SpmmCsrInner(MatrixIt matrix_global, ColsIt cols_global, int nz,
	CsrIt csr_global, SourcesIt sources_global, int numRows, 
	const int* numRows2_global, VecIt vec_global, DestIt dest_global,
	T identity, MulOp mulOp, AddOp addOp, const int B_ncols, LimIt limits_global,
  DestIt carryin_global, DestIt carryout_global, CudaContext& context) {

	int2 launch = Tuning::GetLaunchParams(context);
	int NV = MGPU_NV;

  dim3 nt, nb;
  nt.x = MGPU_NTX;
  nt.y = MGPU_NTY;
  nt.z = MGPU_NTZ;
	nb.x = MGPU_DIV_UP(nz, NV);
  nb.y = 1;
  nb.z = B_ncols/32;

	// Use upper-bound binary search to partition the CSR structure into tiles.
	PartitionCsrSegReducePrealloc(nz, NV, csr_global, numRows, numRows2_global, 
      nb.x + 1, limits_global, context);
		
	// Evaluate the Spmv product.
	//MGPU_MEM(T) carryOutDevice = context.Malloc<T>(numBlocks*MGPU_BC);
	KernelSpmmCsr<Tuning, Indirect, LoadLeft>
		<<<nb, nt, 0, context.Stream()>>>(matrix_global,
		cols_global, nz, csr_global, sources_global, vec_global, 
		limits_global, dest_global, carryin_global, identity, 
		mulOp, addOp, B_ncols);
	MGPU_SYNC_CHECK("KernelSpmmCsr");

  //PrintArray(*limitsDevice, "%4d", 10);
  //printDense(numRows, B_ncols, dest_global);
  //printDense(B_ncols, numBlocks, carryOutDevice->get());

	// Add the carry-in values.
	//SegReduceSpinePrealloc(limits_global, numBlocks, dest_global,
	//	carryin_global, carryout_global, identity, addOp, context);
}

template<typename Tuning, bool Indirect, bool LoadLeft, typename MatrixIt,
	typename ColsIt, typename CsrIt, typename SourcesIt, typename VecIt,
  typename LimIt, typename DestIt, typename T, typename MulOp, typename AddOp>
MGPU_HOST void SpmmCsrHost(MatrixIt matrix_global, ColsIt cols_global, int nz,
	CsrIt csr_global, SourcesIt sources_global, int numRows, VecIt vec_global,
	bool supportEmpty, DestIt dest_global, T identity, MulOp mulOp, AddOp addOp,
	const int B_ncols, LimIt limits_global, DestIt carryin_global, 
  DestIt carryout_global, CudaContext& context) {
		
	if(supportEmpty) {
    std::cout << "Error: supportEmpty is not implemented\n";
	} else {
		SpmmCsrInner<Tuning, Indirect, LoadLeft>(matrix_global, cols_global, nz,
			csr_global, sources_global, numRows, (const int*)0, vec_global, 
			dest_global, identity, mulOp, addOp, B_ncols, limits_global, 
      carryin_global, carryout_global, context);
	}
}

////////////////////////////////////////////////////////////////////////////////
// Spmv host functions

template<typename T>
struct spmv_pass_through : public std::binary_function<T, T, T> {
	MGPU_HOST_DEVICE T operator()(T a, T b) { return a; }
};

template<typename MatrixIt, typename ColsIt, typename CsrIt, typename VecIt,
	typename LimIt, typename DestIt, typename T, typename MulOp, typename AddOp>
MGPU_HOST void SpmmCsrBinary(MatrixIt matrix_global, ColsIt cols_global, int nz,
	CsrIt csr_global, int numRows, VecIt vec_global, bool supportEmpty, 
	DestIt dest_global, T identity, MulOp mulOp, AddOp addOp, const int B_ncols,
  LimIt limits_global, DestIt carryin_global, DestIt carryout_global,
	CudaContext& context) {
			
	typedef typename SpmmTuningNormal<sizeof(T), true>::Tuning Tuning;
	SpmmCsrHost<Tuning, false, true>(matrix_global, cols_global, nz, csr_global,
		(const int*)0, numRows, vec_global, supportEmpty, dest_global, 
		identity, mulOp, addOp, B_ncols, limits_global, carryin_global, 
    carryout_global, context);
}

} // namespace mgpu
