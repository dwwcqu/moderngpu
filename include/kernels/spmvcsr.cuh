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
 *       names of its contributors may be used to endorse or promote products
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

//#include "../constants.h"

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
		SegReduceTuning<128, 7, 0, true, false>
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
	template<int MGPU_TB, typename MatrixIt, typename ColumnsIt, typename VecIt>
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
      //if( data[ii]!=0.f && blockIdx.x==1 && blockIdx.z==0 )
      //  printf("tid %d: %f\n", tid, data[ii]);
    }

		// Clear out the out-of-range inputs. 
		if( count2 < NV && (tid>>5) >= ((count2+31)>>5) )
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

template<int MGPU_TB, int NT, typename Tuning, bool Indirect, bool LoadLeft, 
  typename MatrixIt, 
	typename ColsIt, typename CsrIt, typename SourcesIt, typename VecIt, 
	typename DestIt, typename T, typename MulOp, typename AddOp>
MGPU_LAUNCH_BOUNDS void KernelSpmmCsr(MatrixIt matrix_global,
	ColsIt cols_global, int nz, CsrIt csr_global, SourcesIt sources_global, 
	VecIt vec_global, CsrIt limits_global, DestIt dest_global,
	T* carryOut_global, T identity, MulOp mulOp, AddOp addOp, const int B_ncols) {

	typedef MGPU_LAUNCH_PARAMS Params;
	const int VT = 1;
	const int NV = NT * VT;
	const bool HalfCapacity = (sizeof(T) > sizeof(int)) && Params::HalfCapacity;

	typedef CTAReduce<NT, AddOp> FastReduce;
	typedef CTASegReduce<NT, MGPU_TB, HalfCapacity, T, AddOp> SegReduce;
  typedef CTASpmmLoad<NT, VT, LoadLeft, HalfCapacity, T, MulOp> SpmmLoad;

	union Shared {
		int csr[NV + 1];
	};
	__shared__ Shared shared;
  __shared__ int    shared_csr2[NT>>4]; // 2x number of warps
  __shared__ float  shared_storage[NT];

	int tid = threadIdx.x;
	int block = blockIdx.x;
	int gid = NV * block;
	int count2 = min(NV, nz - gid);
  int lane_id = tid & (32 - 1);
  int warp_id = tid>>5;

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
    columns[0]    = __ldg(cols_global+gid+tid)<<6;
    matrixData[0] = __ldg(matrix_global+gid+tid);
    if( blockIdx.x==1 && blockIdx.z==0 )
      printf("count2:%d,tid:%d,col:%d,val:%f\n", count2, tid, columns[0]>>6, matrixData[0]);
  }
  else
  {
    columns[0]    = 0;
    matrixData[0] = 0.f;
  }

  T carryIn = 0.f;
  T carryOut;

  //for( int slab=0; slab<4; slab+=MGPU_TB )
  for( int slab=0; slab<32; slab+=MGPU_TB )
  {
    // Removed Indirect load case
    // This is a direct load so we don't have a data-dependency on the
    // limits.
    SpmmLoad::LoadDirectSpmm<MGPU_TB>(count2, tid,
        matrixData, columns, vec_global+lane_id+(blockIdx.z<<5), 
        slab, identity, mulOp, data);
    //if( threadIdx.x==0 && blockIdx.z==0 ) printf("%d:%d,%d,%d,%d,%d\n", blockIdx.x, shared_csr2[0], shared_csr2[1], shared_csr2[2], shared_csr2[3], shared_csr2[4]);
      //if( (tid%32) < 16 && tid<48 && blockIdx.z==0 && blockIdx.x==0 )
      //  printf("tid %d: %f,%f,%f,%f\n", tid, data[0], data[1], data[2], data[3]);

    // Flatten CSR->COO and return the segmented scan terms.
    //terms = DeviceSegReducePrepare<NT, MGPU_TB>(shared.csr,
    //    numRows, tid, gid, range.flushLast, rows, rowStarts);
    terms = DeviceSegReducePrepareSpmm<NT, MGPU_TB>(shared.csr, shared_csr2,
        numRows, warp_id<<5, tid, gid, range.flushLast, rows, rowStarts);
    /*if( (lane_id==0 || (lane_id<16 && threadIdx.x<64)) && blockIdx.z==0 )
    {
      printf("tid:%d,row:%d,%d,%d,%d,%d delta:%d\n", tid,rows[0],rows[1],rows[2],rows[3],rows[4],terms.tidDelta);
      printf("tid:%d,rowStart:%d,%d,%d,%d\n", tid,rowStarts[0],rowStarts[1],rowStarts[2],rowStarts[3]);
    }*/
    #pragma unroll
    for( int i=0; i<MGPU_TB+1; i++ )
      rows[i] = __shfl(rows[i], slab/MGPU_TB);
    #pragma unroll
    for( int i=0; i<MGPU_TB; i++ )
      rowStarts[i] = __shfl(rowStarts[i], slab/MGPU_TB);
    terms.tidDelta = __shfl(terms.tidDelta, slab/MGPU_TB);
    /*if( (lane_id==0 || (lane_id<16 && threadIdx.x<64)) && blockIdx.z==0 && blockIdx.x==1 )
    {
      printf("tid:%d,row:%d,%d,%d,%d,%d delta:%d\n", tid,rows[0],rows[1],rows[2],rows[3],rows[4],terms.tidDelta);
      printf("tid:%d,rowStart:%d,%d,%d,%d\n", tid,rowStarts[0],rowStarts[1],rowStarts[2],rowStarts[3]);
    }*/

    // Reduce tile data and store to dest_global. Write tile's carry-out
    // term to carryOut_global.
    carryOut = SegReduce::ReduceToGlobalSpmm(rows, range.total, terms.tidDelta, 
        range.begin, block, tid, lane_id, data, 
        dest_global+(blockIdx.z<<5), carryOut_global+(blockIdx.z<<5), 
        carryIn, slab, identity, addOp, shared_storage);
    carryIn  = carryOut;
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
  DestIt carryin_global, DestIt carryout_global, const int tb, const int nt,
  CudaContext& context) {

	int2 launch = Tuning::GetLaunchParams(context);
	int NV = nt;

  dim3 mgpu_nt, mgpu_nb;
  mgpu_nt.x = nt;
  mgpu_nt.y = 1;
  mgpu_nt.z = 1;
	mgpu_nb.x = MGPU_DIV_UP(nz, NV);
  mgpu_nb.y = 1;
  mgpu_nb.z = B_ncols/32;

	// Use upper-bound binary search to partition the CSR structure into tiles.
	PartitionCsrSegReducePrealloc(nz, NV, csr_global, numRows, numRows2_global, 
      mgpu_nb.x + 1, limits_global, context);
		
	// Evaluate the Spmv product.
	//MGPU_MEM(T) carryOutDevice = context.Malloc<T>(numBlocks*MGPU_BC);
  printf("NT:%d, TB:%d\n",nt,tb);
  switch( tb )
	{
    case( 4 ):
      switch( nt )
      {
        case( 32 ):
        KernelSpmmCsr<4, 32, Tuning, Indirect, LoadLeft>
          <<<mgpu_nb, mgpu_nt, 0, context.Stream()>>>(matrix_global,
          cols_global, nz, csr_global, sources_global, vec_global, 
          limits_global, dest_global, carryin_global, identity, 
          mulOp, addOp, B_ncols);
          break;
        case( 64 ):
        KernelSpmmCsr<4, 64, Tuning, Indirect, LoadLeft>
          <<<mgpu_nb, mgpu_nt, 0, context.Stream()>>>(matrix_global,
          cols_global, nz, csr_global, sources_global, vec_global, 
          limits_global, dest_global, carryin_global, identity, 
          mulOp, addOp, B_ncols);
          break;
        case( 128 ):
        KernelSpmmCsr<4, 128, Tuning, Indirect, LoadLeft>
          <<<mgpu_nb, mgpu_nt, 0, context.Stream()>>>(matrix_global,
          cols_global, nz, csr_global, sources_global, vec_global, 
          limits_global, dest_global, carryin_global, identity, 
          mulOp, addOp, B_ncols);
          break;
        case( 256 ):
        KernelSpmmCsr<4, 256, Tuning, Indirect, LoadLeft>
          <<<mgpu_nb, mgpu_nt, 0, context.Stream()>>>(matrix_global,
          cols_global, nz, csr_global, sources_global, vec_global, 
          limits_global, dest_global, carryin_global, identity, 
          mulOp, addOp, B_ncols);
          break;
        case( 512 ):
        KernelSpmmCsr<4, 512, Tuning, Indirect, LoadLeft>
          <<<mgpu_nb, mgpu_nt, 0, context.Stream()>>>(matrix_global,
          cols_global, nz, csr_global, sources_global, vec_global, 
          limits_global, dest_global, carryin_global, identity, 
          mulOp, addOp, B_ncols);
          break;
        case( 1024 ):
        KernelSpmmCsr<4, 1024, Tuning, Indirect, LoadLeft>
          <<<mgpu_nb, mgpu_nt, 0, context.Stream()>>>(matrix_global,
          cols_global, nz, csr_global, sources_global, vec_global, 
          limits_global, dest_global, carryin_global, identity, 
          mulOp, addOp, B_ncols);
          break;
       }
      break;
    case( 8 ):
      switch( nt )
      {
        case( 32 ):
        KernelSpmmCsr<8, 32, Tuning, Indirect, LoadLeft>
          <<<mgpu_nb, mgpu_nt, 0, context.Stream()>>>(matrix_global,
          cols_global, nz, csr_global, sources_global, vec_global, 
          limits_global, dest_global, carryin_global, identity, 
          mulOp, addOp, B_ncols);
        case( 64 ):
        KernelSpmmCsr<8, 64, Tuning, Indirect, LoadLeft>
          <<<mgpu_nb, mgpu_nt, 0, context.Stream()>>>(matrix_global,
          cols_global, nz, csr_global, sources_global, vec_global, 
          limits_global, dest_global, carryin_global, identity, 
          mulOp, addOp, B_ncols);
          break;
        case( 128 ):
        KernelSpmmCsr<8, 128, Tuning, Indirect, LoadLeft>
          <<<mgpu_nb, mgpu_nt, 0, context.Stream()>>>(matrix_global,
          cols_global, nz, csr_global, sources_global, vec_global, 
          limits_global, dest_global, carryin_global, identity, 
          mulOp, addOp, B_ncols);
          break;
        case( 256 ):
        KernelSpmmCsr<8, 256, Tuning, Indirect, LoadLeft>
          <<<mgpu_nb, mgpu_nt, 0, context.Stream()>>>(matrix_global,
          cols_global, nz, csr_global, sources_global, vec_global, 
          limits_global, dest_global, carryin_global, identity, 
          mulOp, addOp, B_ncols);
          break;
        case( 512 ):
        KernelSpmmCsr<8, 512, Tuning, Indirect, LoadLeft>
          <<<mgpu_nb, mgpu_nt, 0, context.Stream()>>>(matrix_global,
          cols_global, nz, csr_global, sources_global, vec_global, 
          limits_global, dest_global, carryin_global, identity, 
          mulOp, addOp, B_ncols);
          break;
        case( 1024 ):
        KernelSpmmCsr<8, 1024, Tuning, Indirect, LoadLeft>
          <<<mgpu_nb, mgpu_nt, 0, context.Stream()>>>(matrix_global,
          cols_global, nz, csr_global, sources_global, vec_global, 
          limits_global, dest_global, carryin_global, identity, 
          mulOp, addOp, B_ncols);
          break;
      }
      break;
    case( 16 ):
      switch( nt )
      {
        case( 32 ):
        KernelSpmmCsr<16, 32, Tuning, Indirect, LoadLeft>
          <<<mgpu_nb, mgpu_nt, 0, context.Stream()>>>(matrix_global,
          cols_global, nz, csr_global, sources_global, vec_global, 
          limits_global, dest_global, carryin_global, identity, 
          mulOp, addOp, B_ncols);
        case( 64 ):
        KernelSpmmCsr<16, 64, Tuning, Indirect, LoadLeft>
          <<<mgpu_nb, mgpu_nt, 0, context.Stream()>>>(matrix_global,
          cols_global, nz, csr_global, sources_global, vec_global, 
          limits_global, dest_global, carryin_global, identity, 
          mulOp, addOp, B_ncols);
          break;
        case( 128 ):
        KernelSpmmCsr<16, 128, Tuning, Indirect, LoadLeft>
          <<<mgpu_nb, mgpu_nt, 0, context.Stream()>>>(matrix_global,
          cols_global, nz, csr_global, sources_global, vec_global, 
          limits_global, dest_global, carryin_global, identity, 
          mulOp, addOp, B_ncols);
          break;
        case( 256 ):
        KernelSpmmCsr<16, 256, Tuning, Indirect, LoadLeft>
          <<<mgpu_nb, mgpu_nt, 0, context.Stream()>>>(matrix_global,
          cols_global, nz, csr_global, sources_global, vec_global, 
          limits_global, dest_global, carryin_global, identity, 
          mulOp, addOp, B_ncols);
          break;
        case( 512 ):
        KernelSpmmCsr<16, 512, Tuning, Indirect, LoadLeft>
          <<<mgpu_nb, mgpu_nt, 0, context.Stream()>>>(matrix_global,
          cols_global, nz, csr_global, sources_global, vec_global, 
          limits_global, dest_global, carryin_global, identity, 
          mulOp, addOp, B_ncols);
          break;
        case( 1024 ):
        KernelSpmmCsr<16, 1024, Tuning, Indirect, LoadLeft>
          <<<mgpu_nb, mgpu_nt, 0, context.Stream()>>>(matrix_global,
          cols_global, nz, csr_global, sources_global, vec_global, 
          limits_global, dest_global, carryin_global, identity, 
          mulOp, addOp, B_ncols);
          break;
      }
      break;
    case( 32 ):
      switch( nt )
      {
        case( 32 ):
        KernelSpmmCsr<32, 32, Tuning, Indirect, LoadLeft>
          <<<mgpu_nb, mgpu_nt, 0, context.Stream()>>>(matrix_global,
          cols_global, nz, csr_global, sources_global, vec_global, 
          limits_global, dest_global, carryin_global, identity, 
          mulOp, addOp, B_ncols);
        case( 64 ):
        KernelSpmmCsr<32, 64, Tuning, Indirect, LoadLeft>
          <<<mgpu_nb, mgpu_nt, 0, context.Stream()>>>(matrix_global,
          cols_global, nz, csr_global, sources_global, vec_global, 
          limits_global, dest_global, carryin_global, identity, 
          mulOp, addOp, B_ncols);
          break;
        case( 128 ):
        KernelSpmmCsr<32, 128, Tuning, Indirect, LoadLeft>
          <<<mgpu_nb, mgpu_nt, 0, context.Stream()>>>(matrix_global,
          cols_global, nz, csr_global, sources_global, vec_global, 
          limits_global, dest_global, carryin_global, identity, 
          mulOp, addOp, B_ncols);
          break;
        case( 256 ):
        KernelSpmmCsr<32, 256, Tuning, Indirect, LoadLeft>
          <<<mgpu_nb, mgpu_nt, 0, context.Stream()>>>(matrix_global,
          cols_global, nz, csr_global, sources_global, vec_global, 
          limits_global, dest_global, carryin_global, identity, 
          mulOp, addOp, B_ncols);
          break;
        case( 512 ):
        KernelSpmmCsr<32, 512, Tuning, Indirect, LoadLeft>
          <<<mgpu_nb, mgpu_nt, 0, context.Stream()>>>(matrix_global,
          cols_global, nz, csr_global, sources_global, vec_global, 
          limits_global, dest_global, carryin_global, identity, 
          mulOp, addOp, B_ncols);
          break;
        case( 1024 ):
        KernelSpmmCsr<32, 1024, Tuning, Indirect, LoadLeft>
          <<<mgpu_nb, mgpu_nt, 0, context.Stream()>>>(matrix_global,
          cols_global, nz, csr_global, sources_global, vec_global, 
          limits_global, dest_global, carryin_global, identity, 
          mulOp, addOp, B_ncols);
          break;
      }
      break;
  }
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
  DestIt carryout_global, const int tb, const int nt, CudaContext& context) {
		
	if(supportEmpty) {
    std::cout << "Error: supportEmpty is not implemented\n";
	} else {
		SpmmCsrInner<Tuning, Indirect, LoadLeft>(matrix_global, cols_global, nz,
			csr_global, sources_global, numRows, (const int*)0, vec_global, 
			dest_global, identity, mulOp, addOp, B_ncols, limits_global, 
      carryin_global, carryout_global, tb, nt, context);
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
  const int tb, const int nt, CudaContext& context) {
			
	typedef typename SpmmTuningNormal<sizeof(T), true>::Tuning Tuning;
	SpmmCsrHost<Tuning, false, true>(matrix_global, cols_global, nz, csr_global,
		(const int*)0, numRows, vec_global, supportEmpty, dest_global, 
		identity, mulOp, addOp, B_ncols, limits_global, carryin_global, 
    carryout_global, tb, nt, context);
}

} // namespace mgpu
