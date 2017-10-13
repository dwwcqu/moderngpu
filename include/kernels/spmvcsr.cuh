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

template<size_t Size, bool LoadLeft>
struct SpmvTuningIndirect {
	enum { Indirect = true };
	typedef LaunchBox<
		SegReduceTuning<128, 11, 0, false, false>,
		SegReduceTuning<128, 11, 0, true, false>,
		SegReduceTuning<128, 7, 0, true, false>
	> Tuning;
};

template<size_t Size, bool LoadLeft>
struct SpmvTuningPreprocess {
	enum { Indirect = false };
	typedef LaunchBox<
		SegReduceTuning<128, 11, 0, false, false>,
		SegReduceTuning<128, 11, 0, true, false>,
		SegReduceTuning<128, (Size > 4) ? 11 : 7, 0, true, false>
	> Tuning;
};


////////////////////////////////////////////////////////////////////////////////
// CTASpmvLoad
// Loads matrix values and column indices and gathers vector values. Finds 
// products and transposes terms into register output in thread order.

template<int NT, int VT, bool LoadLeft, bool HalfCapacity, typename T,
	typename MulOp>
struct CTASpmvLoad {
	enum {
		NV = NT * VT,
		Capacity = HalfCapacity ? (NV / 2) : NV
	};
	//typedef CTASegReduce<NT, VT, HalfCapacity, T, MulOp> SegReduce;
	
	union Storage {
		int sources[NV];
		T data[Capacity];
		//typename SegReduce::Storage segReduceStorage;
	};

	template<typename MatrixIt, typename ColumnsIt, typename VecIt>
	MGPU_DEVICE static void LoadDirect(int count2, int tid, int gid, 
		MatrixIt matrix_global, ColumnsIt cols_global, VecIt vec_global, 
		T identity, MulOp mulOp, T data[VT], Storage& storage) {

		// Load columns directly from cols_global.
		int columns[VT];
		DeviceGlobalToRegDefault<NT, VT>(count2, cols_global + gid, tid,
			columns, 0);

		// Load data into stridedData.
		T matrixData[VT];
		if(LoadLeft)
			DeviceGlobalToRegDefault<NT, VT>(count2, matrix_global + gid,
				tid, matrixData, identity);
		
		// Use ldg to load vector data in strided order.
		T vecData[VT];
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			vecData[i] = ldg(vec_global + columns[i]);
		
		// Clear out the out-of-range inputs. 
		if(count2 < NV) {
			#pragma unroll
			for(int i = 0; i < VT; ++i)
				if(NT * i + tid >= count2)
					vecData[i] = identity;
		}	

		// Multiply matrix and vector values together.
		T stridedData[VT];
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			stridedData[i] = LoadLeft ? 
				mulOp(matrixData[i], vecData[i]) : vecData[i];

		// Transpose from strided to thread order.
		if(HalfCapacity)
			HalfSmemTranspose<NT, VT>(stridedData, tid, storage.data, data);
		else {
			DeviceRegToShared<NT, VT>(stridedData, tid, storage.data);
			DeviceSharedToThread<VT>(storage.data, tid, data);
		}
	}

	template<typename MatrixIt, typename ColumnsIt, typename VecIt>
	MGPU_DEVICE static void LoadDirectSpmm(int count2, int tid, int gid, 
		MatrixIt matrix_global, ColumnsIt cols_global, VecIt vec_global, 
		T identity, MulOp mulOp, T data[VT*MGPU_TB], Storage& storage) {

    const int NT2 = MGPU_NTX;

		// Load column indices directly from cols_global.
		int columns[VT];
		DeviceGlobalToRegDefault<NT2, VT>(count2, cols_global + gid, tid,
			columns, 0);

		// Load values into stridedData.
		T matrixData[VT];
		if(LoadLeft)
			DeviceGlobalToRegDefault<NT2, VT>(count2, matrix_global + gid,
				tid, matrixData, identity);
		
		// Use ldg to load vector data in strided order.
		T      vecData[VT*MGPU_TB];
		#pragma unroll
		for(int i = 0; i < VT; ++i)
		{
      #pragma unroll
      for( int j=0; j<MGPU_TB; j++ )
    	  vecData[i*MGPU_TB+j] = ldg(vec_global + columns[i]*MGPU_BC + j);
    }

		// Clear out the out-of-range inputs. 
		if(count2 < NT2*VT)
    {
			#pragma unroll
			for(int i = 0; i < VT; ++i)
				if(NT2 * i + tid >= count2)
				{
        	#pragma unroll
          for( int j=0; j<MGPU_TB; j++ )
            vecData[i*MGPU_TB+j] = identity;
        }
		}	

		// Multiply matrix and vector values together.
		T stridedData[VT*MGPU_TB];
		#pragma unroll
		for(int i = 0; i < VT; ++i)
		{
      #pragma unroll
      for( int j=0; j<MGPU_TB; j++ )
      {
        stridedData[i+j*VT] = LoadLeft ? 
          mulOp(matrixData[i], vecData[i*MGPU_TB+j]) : vecData[i*MGPU_TB+j];
        //if( stridedData[i+j*VT]!=0.f && blockIdx.x==1 )
        //  printf("%d,%d,%d,%d,%d:%f\n", tid, i,j,i+j*VT, columns[i], stridedData[i+j*VT]); 
      }
    }

    const int tiy = threadIdx.y;
		// Transpose from strided to thread order.
		if(HalfCapacity)
			HalfSmemTranspose<NT, VT*MGPU_TB>(stridedData, tid, storage.data, data);
		else {
      // Cannot unroll, because using smem resource sequentially
      for( int j=0; j<MGPU_TB; j++ )
      {
			  DeviceRegToShared<NT2, VT>(stridedData+j*VT, tid, storage.data+NV*tiy);
			  DeviceSharedToThread<VT>(storage.data+NV*tiy, tid, data+j*VT);
      }
		}
	}

	template<typename MatrixIt, typename ColumnsIt, typename VecIt>
	MGPU_DEVICE static void LoadDirectSpmmVector(int count2, int tid, int gid, 
		MatrixIt matrix_global, ColumnsIt cols_global, VecIt vec_global, 
		T identity, MulOp mulOp, T data[VT*MGPU_TB], Storage& storage) {

		// Load column indices directly from cols_global.
		int columns[VT];
		DeviceGlobalToRegDefault<NT, VT>(count2, cols_global + gid, tid,
			columns, 0);

		// Load values into stridedData.
		T matrixData[VT];
		if(LoadLeft)
			DeviceGlobalToRegDefault<NT, VT>(count2, matrix_global + gid,
				tid, matrixData, identity);
		
		// Use ldg to load vector data in strided order.
    float4 rawData[VT*(MGPU_TB>>2)];
		#pragma unroll
		for(int i = 0; i < VT; ++i)
		{
      #pragma unroll
      for( int j=0; j<MGPU_TB>>2; j++ )
        rawData[i*(MGPU_TB>>2)+j] = ldg((float4*)(vec_global+columns[i]*MGPU_BC+
            j*4));
    }

		// Clear out the out-of-range inputs. 
		if(count2 < NV)
    {
			#pragma unroll
			for(int i = 0; i < VT; ++i)
				if(NT * i + tid >= count2)
				{
        	#pragma unroll
          for( int j=0; j<MGPU_TB>>2; j++ )
          {
            rawData[i*(MGPU_TB>>2)+j].x = identity;
            rawData[i*(MGPU_TB>>2)+j].y = identity;
            rawData[i*(MGPU_TB>>2)+j].z = identity;
            rawData[i*(MGPU_TB>>2)+j].w = identity;
          }
        }
		}	

		// Multiply matrix and vector values together.
		T stridedData[VT*MGPU_TB];
		#pragma unroll
		for(int i = 0; i < VT; ++i)
		{
      #pragma unroll
      for( int j=0; j<MGPU_TB>>2; j++ )
      {
        //if( !tid ) printf("%d,%d\n", i, j);
        stridedData[i+j*4*VT  ] = LoadLeft ? 
          mulOp(matrixData[i], rawData[i*(MGPU_TB>>2)+j].x) : rawData[i*(MGPU_TB>>2)+j].x;
        stridedData[i+j*4*VT+1*VT] = LoadLeft ? 
          mulOp(matrixData[i], rawData[i*(MGPU_TB>>2)+j].y) : rawData[i*(MGPU_TB>>2)+j].y;
        stridedData[i+j*4*VT+2*VT] = LoadLeft ? 
          mulOp(matrixData[i], rawData[i*(MGPU_TB>>2)+j].z) : rawData[i*(MGPU_TB>>2)+j].z;
        stridedData[i+j*4*VT+3*VT] = LoadLeft ? 
          mulOp(matrixData[i], rawData[i*(MGPU_TB>>2)+j].w) : rawData[i*(MGPU_TB>>2)+j].w;
        //stridedData[i*MGPU_TB+j] = LoadLeft ? 
        //  mulOp(matrixData[i], vecData[i*MGPU_TB+j]) : vecData[i*MGPU_TB+j];
        //if( stridedData[i+j*VT]!=0.f )
        //  printf("%d,%d,%d,%d,%d:%f\n", tid, i,j,i+j*VT, columns[i], stridedData[i+j*VT]); 
      }
    }

    const int tiy = threadIdx.y;
		// Transpose from strided to thread order.
		if(HalfCapacity)
			HalfSmemTranspose<NT, VT*MGPU_TB>(stridedData, tid, storage.data, data);
		else {
      // Cannot unroll, because using smem resource sequentially
      for( int j=0; j<MGPU_TB; j++ )
      {
			  DeviceRegToShared<NT, VT>(stridedData+j*VT, tid, storage.data+tiy*NV);
			  DeviceSharedToThread<VT>(storage.data+tiy*NV, tid, data+j*VT);
      }
		}
	}
	template<typename SourcesIt, typename MatrixIt, typename ColumnsIt,
		typename VecIt>
	MGPU_DEVICE static void LoadIndirect(int count2, int tid, int gid, 
		int numRows, int startRow, const int rows[VT], const int rowStarts[VT],
		SourcesIt sources_global, MatrixIt matrix_global, 
		ColumnsIt cols_global, VecIt vec_global, T identity, MulOp mulOp,
		T data[VT], Storage& storage) {
			
		// Load source offsets from sources_global into smem.
		DeviceGlobalToSharedLoop<NT, VT>(numRows, sources_global + startRow,
			tid, storage.sources);

		// Compute the offset of each element within its row.
		int indices[VT];
		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			int index = VT * tid + i;
			int rowOffset = gid + index - rowStarts[i];
			int source = storage.sources[rows[i]];
			indices[i] = source + rowOffset;
		}
		__syncthreads();

		// Transpose indices through shared memory into strided order.
		DeviceThreadToShared<VT>(indices, tid, storage.sources);
		DeviceSharedToReg<NT, VT>(storage.sources, tid, indices);

		// Gather columns from cols_global.
		int columns[VT];
		DeviceGatherDefault<NT, VT>(count2, cols_global, indices, tid, 
			columns, 0);

		// Gather data into stridedData.
		T matrixData[VT];
		if(LoadLeft)
			DeviceGatherDefault<NT, VT>(count2, matrix_global, indices, 
				tid, matrixData, identity);						
		
		// Use ldg to load vector data in strided order.
		T vecData[VT];
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			vecData[i] = ldg(vec_global + columns[i]);

		// Multiply matrix and vector values together.
		T stridedData[VT];
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			stridedData[i] = LoadLeft ? 
				mulOp(matrixData[i], vecData[i]) : vecData[i];

		// Transpose from strided to thread order.
		if(HalfCapacity)
			HalfSmemTranspose<NT, VT>(stridedData, tid, storage.data, data);
		else {
			DeviceRegToShared<NT, VT>(stridedData, tid, storage.data);
			DeviceSharedToThread<VT>(storage.data, tid, data);
		}
	}
};

////////////////////////////////////////////////////////////////////////////////
// KernelSpmvCsr

template<typename Tuning, bool Indirect, bool LoadLeft, typename MatrixIt, 
	typename ColsIt, typename CsrIt, typename SourcesIt, typename VecIt, 
	typename DestIt, typename T, typename MulOp, typename AddOp>
MGPU_LAUNCH_BOUNDS void KernelSpmvCsr(MatrixIt matrix_global,
	ColsIt cols_global, int nz, CsrIt csr_global, SourcesIt sources_global, 
	VecIt vec_global, const int* limits_global, DestIt dest_global,
	T* carryOut_global, T identity, MulOp mulOp, AddOp addOp) {

	typedef MGPU_LAUNCH_PARAMS Params;
	const int NT = Params::NT;
	const int VT = Params::VT;
	const int NV = NT * VT;
	const bool HalfCapacity = (sizeof(T) > sizeof(int)) && Params::HalfCapacity;

	typedef CTAReduce<NT, AddOp> FastReduce;
	typedef CTASegReduce<NT, VT, HalfCapacity, T, AddOp> SegReduce;
	typedef CTASpmvLoad<NT, VT, LoadLeft, HalfCapacity, T, MulOp> SpmvLoad;

	union Shared {
		int csr[NV + 1];
		typename SegReduce::Storage segReduceStorage;
		typename SpmvLoad::Storage spmvLoadStorage;
	};
	__shared__ Shared shared;

	int tid = threadIdx.x;
	int block = blockIdx.x;
	int gid = NV * block;
	int count2 = min(NV, nz - gid);

	// Retrieve the left and right row limits.
	int limit0 = limits_global[block];
	int limit1 = limits_global[block + 1];

	SegReduceRange range;
	SegReduceTerms terms;
	int rows[VT + 1], rowStarts[VT];
	T data[VT];

	if(Indirect) {
		// Transform the row limits into ranges.
		range = DeviceShiftRange(limit0, limit1);
		int numRows = range.end - range.begin;

		// Load the CSR interval.
		DeviceGlobalToSharedLoop<NT, VT>(numRows, csr_global + range.begin, tid, 
			shared.csr);

		// Flatten CSR->COO and return the segmented scan terms.
		terms = DeviceSegReducePrepare<NT, VT>(shared.csr, numRows, tid, gid, 
			range.flushLast, rows, rowStarts);

		// Load tile of data in thread order from row IDs.
		SpmvLoad::LoadIndirect(count2, tid, gid, numRows, range.begin, rows, 
			rowStarts, sources_global, matrix_global, cols_global, vec_global,
			identity, mulOp, data, shared.spmvLoadStorage);
	} else {
		// This is a direct load so we don't have a data-dependency on the
		// limits.
		SpmvLoad::LoadDirect(count2, tid, gid, matrix_global, cols_global,
			vec_global, identity, mulOp, data, shared.spmvLoadStorage);

		// Transform the row limits into ranges.
		range = DeviceShiftRange(limit0, limit1);
		int numRows = range.end - range.begin;

		// Load the CSR interval.
		DeviceGlobalToSharedLoop<NT, VT>(numRows, csr_global + range.begin, tid, 
			shared.csr);

		// Flatten CSR->COO and return the segmented scan terms.
		terms = DeviceSegReducePrepare<NT, VT>(shared.csr, numRows, tid, gid,
			range.flushLast, rows, rowStarts);
	}

	// Reduce tile data and store to dest_global. Write tile's carry-out
	// term to carryOut_global.
	SegReduce::ReduceToGlobal(rows, range.total, terms.tidDelta, 
		range.begin, block, tid, data, dest_global, carryOut_global,
		identity, addOp, shared.segReduceStorage);
}

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
  const int NT2= MGPU_NTX;
  const int NV2= MGPU_NTX * VT;
	const bool HalfCapacity = (sizeof(T) > sizeof(int)) && Params::HalfCapacity;

	typedef CTAReduce<NT, AddOp> FastReduce;
	typedef CTASegReduce<NT, VT, HalfCapacity, T, AddOp> SegReduce;
	typedef CTASpmvLoad<NT, VT, LoadLeft, HalfCapacity, T, MulOp> SpmvLoad;

	union Shared {
		int csr[NV2 + 1];
		typename SegReduce::Storage segReduceStorage;
		typename SpmvLoad::Storage spmvLoadStorage;
	};
	__shared__ Shared shared;

	int tid = threadIdx.x;
	int block = blockIdx.x;
	int gid = NV2 * block;
	int count2 = min(NV2, nz - gid);

	// Retrieve the left and right row limits.
	int limit0 = limits_global[block];
	int limit1 = limits_global[block + 1];

	SegReduceRange range;
	SegReduceTerms terms;
	int rows[VT + 1], rowStarts[VT];
	T data[VT*MGPU_TB];

  // Transform the row limits into ranges.
  range = DeviceShiftRange(limit0, limit1);
  int numRows = range.end - range.begin;

  //if( tid==0 ) printf("%d,%d,%d,%d\n", block, limit0, limit1, numRows);

  // Load the CSR interval.
  if( threadIdx.y==0 )
    DeviceGlobalToSharedLoop<NT2, VT>(numRows, csr_global + range.begin, tid, 
      shared.csr);
  __syncthreads();

  // Flatten CSR->COO and return the segmented scan terms.
  terms = DeviceSegReducePrepare<NT2, VT>(shared.csr, numRows, tid, gid,
    range.flushLast, rows, rowStarts);

  int tiy = threadIdx.y + blockIdx.y*blockDim.y;
  //for( int slab=0; slab<1; slab++ )
  //{
    // Removed Indirect load case
    // This is a direct load so we don't have a data-dependency on the
    // limits.
    SpmvLoad::LoadDirectSpmm(count2, tid, gid, 
      matrix_global, cols_global, vec_global+tiy*MGPU_TB, 
      identity, mulOp, data, shared.spmvLoadStorage);

  // Reduce tile data and store to dest_global. Write tile's carry-out
    // term to carryOut_global.
    SegReduce::ReduceToGlobalSpmm(rows, range.total, terms.tidDelta, 
		  range.begin, block, tid, data, dest_global+tiy*MGPU_TB, 
      carryOut_global+tiy*MGPU_TB*gridDim.x,
		  identity, addOp, shared.segReduceStorage);
  //}
}

////////////////////////////////////////////////////////////////////////////////
// SpmvCsrHost

template<typename Tuning, bool Indirect, bool LoadLeft, typename MatrixIt, 
	typename ColsIt, typename CsrIt, typename SourcesIt, typename VecIt,
	typename DestIt, typename T, typename MulOp, typename AddOp>
MGPU_HOST void SpmvCsrInner(MatrixIt matrix_global, ColsIt cols_global, int nz,
	CsrIt csr_global, SourcesIt sources_global, int numRows, 
	const int* numRows2_global, VecIt vec_global, DestIt dest_global,
	T identity, MulOp mulOp, AddOp addOp, CudaContext& context) {

	int2 launch = Tuning::GetLaunchParams(context);
	int NV = launch.x * launch.y;

	int numBlocks = MGPU_DIV_UP(nz, NV);

	// Use upper-bound binary search to partition the CSR structure into tiles.
	MGPU_MEM(int) limitsDevice = PartitionCsrSegReduce(nz, NV, csr_global,
		numRows, numRows2_global, numBlocks + 1, context);
		
	// Evaluate the Spmv product.
	MGPU_MEM(T) carryOutDevice = context.Malloc<T>(numBlocks);
	KernelSpmvCsr<Tuning, Indirect, LoadLeft>
		<<<numBlocks, launch.x, 0, context.Stream()>>>(matrix_global,
		cols_global, nz, csr_global, sources_global, vec_global, 
		limitsDevice->get(), dest_global, carryOutDevice->get(), identity, 
		mulOp, addOp);
	MGPU_SYNC_CHECK("KernelSpmvCsr");

	// Add the carry-in values.
	SegReduceSpine(limitsDevice->get(), numBlocks, dest_global,
		carryOutDevice->get(), identity, addOp, context);
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
  nt.z = 1;
	nb.x = MGPU_DIV_UP(nz, NV);
  nb.y = MGPU_BC/MGPU_TB/MGPU_NTY;
  nb.z = 1;

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
	typename DestIt, typename T, typename MulOp, typename AddOp>
MGPU_HOST void SpmvCsrHost(MatrixIt matrix_global, ColsIt cols_global, int nz,
	CsrIt csr_global, SourcesIt sources_global, int numRows, VecIt vec_global,
	bool supportEmpty, DestIt dest_global, T identity, MulOp mulOp, AddOp addOp,
  CudaContext& context) {
		
	if(supportEmpty) {
		// Allocate space for CSR2 and Sources2.
		MGPU_MEM(int) csr2Device = context.Malloc<int>(numRows + 1);
		MGPU_MEM(int) sources2Device;
		if(Indirect) sources2Device = context.Malloc<int>(numRows);

		// Strip the empties from CSR and store in CSR2.
		CsrStripEmpties<Indirect>(nz, csr_global, sources_global, numRows,
			csr2Device->get(), Indirect ? sources2Device->get() : (int*)0, 
			(int*)0, context);

		// Run the Spmv in the CSR2 coordinate space.
		MGPU_MEM(T) destDevice = context.Malloc<T>(numRows);
		SpmvCsrInner<Tuning, Indirect, LoadLeft>(matrix_global, cols_global, nz,
			csr2Device->get(), Indirect ? sources2Device->get() : (const int*)0,
			-1, csr2Device->get() + numRows, vec_global,
			destDevice->get(), identity, mulOp, addOp, context);
		
		// Transform into the CSR space with BulkInsert.
		CsrBulkInsert(csr2Device->get(), numRows, destDevice->get(), identity,
			dest_global, context);

	} else {
		SpmvCsrInner<Tuning, Indirect, LoadLeft>(matrix_global, cols_global, nz,
			csr_global, sources_global, numRows, (const int*)0, vec_global, 
			dest_global, identity, mulOp, addOp, context);
	}
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
		// Allocate space for CSR2 and Sources2.
		/*MGPU_MEM(int) csr2Device = context.Malloc<int>(numRows + 1);
		MGPU_MEM(int) sources2Device;
		if(Indirect) sources2Device = context.Malloc<int>(numRows);

		// Strip the empties from CSR and store in CSR2.
		CsrStripEmpties<Indirect>(nz, csr_global, sources_global, numRows,
			csr2Device->get(), Indirect ? sources2Device->get() : (int*)0, 
			(int*)0, context);

		// Run the Spmv in the CSR2 coordinate space.
		MGPU_MEM(T) destDevice = context.Malloc<T>(numRows);
		SpmmCsrInner<Tuning, Indirect, LoadLeft>(matrix_global, cols_global, nz,
			csr2Device->get(), Indirect ? sources2Device->get() : (const int*)0,
			-1, csr2Device->get() + numRows, vec_global,
			destDevice->get(), identity, mulOp, addOp, B_ncols, context);
		
		// Transform into the CSR space with BulkInsert.
		CsrBulkInsert(csr2Device->get(), numRows, destDevice->get(), identity,
			dest_global, context);*/

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

template<typename ColsIt, typename CsrIt, typename VecIt, typename DestIt,
	typename T, typename AddOp>
MGPU_HOST void SpmvCsrUnary( ColsIt cols_global, int nz, CsrIt csr_global,
	int numRows, VecIt vec_global, bool supportEmpty, DestIt dest_global,
	T identity, AddOp addOp, CudaContext& context) {

	typedef typename SpmvTuningNormal<sizeof(T), false>::Tuning Tuning;
	SpmvCsrHost<Tuning, false, false>((const T*)0, cols_global, nz, csr_global,
		(const int*)0, numRows, vec_global, supportEmpty, dest_global, 
		identity, spmv_pass_through<T>(), addOp, context);
}

template<typename MatrixIt, typename ColsIt, typename CsrIt, typename VecIt,
	typename DestIt, typename T, typename MulOp, typename AddOp>
MGPU_HOST void SpmvCsrBinary(MatrixIt matrix_global, ColsIt cols_global, int nz,
	CsrIt csr_global, int numRows, VecIt vec_global, bool supportEmpty, 
	DestIt dest_global, T identity, MulOp mulOp, AddOp addOp, 
	CudaContext& context) {
			
	typedef typename SpmvTuningNormal<sizeof(T), true>::Tuning Tuning;
	SpmvCsrHost<Tuning, false, true>(matrix_global, cols_global, nz, csr_global,
		(const int*)0, numRows, vec_global, supportEmpty, dest_global, 
		identity, mulOp, addOp, context);
}

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

template<typename ColsIt, typename CsrIt, typename SourcesIt, typename VecIt,
	typename DestIt, typename T, typename AddOp>
MGPU_HOST void SpmvCsrIndirectUnary(ColsIt cols_global, int nz,
	CsrIt csr_global, SourcesIt sources_global, int numRows, VecIt vec_global,
	bool supportEmpty, DestIt dest_global, T identity, AddOp addOp, 
	CudaContext& context) {

	typedef typename SpmvTuningIndirect<sizeof(T), false>::Tuning Tuning;
	SpmvCsrHost<Tuning, true, false>((const T*)0, cols_global, nz, csr_global,
		sources_global, numRows, vec_global, supportEmpty, dest_global, 
		identity, spmv_pass_through<T>(), addOp, context);
}

template<typename MatrixIt, typename ColsIt, typename CsrIt, typename SourcesIt,
	typename VecIt, typename DestIt, typename T, typename MulOp, typename AddOp>
MGPU_HOST void SpmvCsrIndirectBinary(MatrixIt matrix_global, ColsIt cols_global,
	int nz, CsrIt csr_global, SourcesIt sources_global, int numRows,
	VecIt vec_global, bool supportEmpty, DestIt dest_global, T identity,
	MulOp mulOp, AddOp addOp, CudaContext& context) {

	typedef typename SpmvTuningIndirect<sizeof(T), true>::Tuning Tuning;
	SpmvCsrHost<Tuning, true, true>(matrix_global, cols_global, nz, csr_global,
		sources_global, numRows, vec_global, supportEmpty, dest_global, 
		identity, mulOp, addOp, context);
}

////////////////////////////////////////////////////////////////////////////////
// Spmv preprocessing

template<typename T, typename CsrIt>
MGPU_HOST void SpmvPreprocessUnary(int nz, CsrIt csr_global, int numRows,
	bool supportEmpty, std::auto_ptr<SpmvPreprocessData>* ppData, 
	CudaContext& context) {

	typedef typename SpmvTuningPreprocess<sizeof(T), false>::Tuning Tuning;
	SegReducePreprocess<Tuning>(nz, csr_global, numRows, supportEmpty, ppData, 
		context);
}

template<typename T, typename CsrIt>
MGPU_HOST void SpmvPreprocessBinary(int nz, CsrIt csr_global, int numRows,
	bool supportEmpty, std::auto_ptr<SpmvPreprocessData>* ppData,
	CudaContext& context) {

	typedef typename SpmvTuningPreprocess<sizeof(T), true>::Tuning Tuning;
	SegReducePreprocess<Tuning>(nz, csr_global, numRows, supportEmpty, ppData, 
		context);
}

////////////////////////////////////////////////////////////////////////////////
// KernelSpmvApply

template<typename Tuning, bool LoadLeft, typename MatrixIt, typename ColsIt,
	typename VecIt, typename DestIt, typename T, typename MulOp, typename AddOp>
MGPU_LAUNCH_BOUNDS void KernelSpmvApply(const int* threadCodes_global,
	MatrixIt matrix_global, ColsIt cols_global, int nz, VecIt vec_global, 
	const int* limits_global, DestIt dest_global, T* carryOut_global, 
	T identity, MulOp mulOp, AddOp addOp) {

	typedef MGPU_LAUNCH_PARAMS Params;
	const int NT = Params::NT;
	const int VT = Params::VT;
	const int NV = NT * VT;
	const bool HalfCapacity = (sizeof(T) > sizeof(int)) && Params::HalfCapacity;

	typedef CTASegReduce<NT, VT, HalfCapacity, T, AddOp> SegReduce;
	typedef CTASpmvLoad<NT, VT, LoadLeft, HalfCapacity, T, MulOp> SpmvLoad;

	union Shared {
		typename SegReduce::Storage segReduceStorage;
		typename SpmvLoad::Storage spmvLoadStorage;
	};
	__shared__ Shared shared;

	int tid = threadIdx.x;
	int block = blockIdx.x;
	int gid = NV * block;
	int count2 = min(NV, nz - gid);

	// Retrieve the left and right row limits and thread codes..
	int limit0 = limits_global[block];
	int limit1 = limits_global[block + 1];
	int threadCodes = threadCodes_global[NT * block + tid];
	
	// Load the tile's data before dereferencing limit0/limit1.
	T data[VT];
	SpmvLoad::LoadDirect(count2, tid, gid, matrix_global, cols_global,
		vec_global, identity, mulOp, data, shared.spmvLoadStorage);

	// Transform the row limits into ranges.
	SegReduceRange range = DeviceShiftRange(limit0, limit1);

	// Expand the row indices.
	int rows[VT + 1];
	DeviceExpandFlagsToRows<VT>(threadCodes>> 20, threadCodes, rows);

	// Reduce tile data and store to dest_global. Write tile's carry-out
	// term to carryOut_global.
	int tidDelta = 0x7f & (threadCodes>> 13);
	SegReduce::ReduceToGlobal(rows, range.total, tidDelta, range.begin,	
		block, tid, data, dest_global, carryOut_global, identity, addOp, 
		shared.segReduceStorage);
}

template<bool LoadLeft, typename MatrixIt, typename ColsIt, typename VecIt, 
	typename DestIt, typename T, typename MulOp, typename AddOp>
MGPU_HOST void SpmvApplyHost(const SpmvPreprocessData& preprocess,
	MatrixIt matrix_global, ColsIt cols_global, VecIt vec_global, 
	DestIt dest_global, T identity, MulOp mulOp, AddOp addOp,
	CudaContext& context) {

	typedef typename SpmvTuningPreprocess<sizeof(T), LoadLeft>::Tuning Tuning;
	int2 launch = Tuning::GetLaunchParams(context);

	if(preprocess.csr2Device.get()) {
		// Support empties.
		MGPU_MEM(T) destDevice = context.Malloc<T>(preprocess.numSegments2);
		MGPU_MEM(T) carryOutDevice = context.Malloc<T>(preprocess.numBlocks);
		KernelSpmvApply<Tuning, LoadLeft>
			<<<preprocess.numBlocks, launch.x, 0, context.Stream()>>>(
			preprocess.threadCodesDevice->get(), matrix_global, cols_global,
			preprocess.count, vec_global, preprocess.limitsDevice->get(),
			destDevice->get(), carryOutDevice->get(), identity, mulOp,
			addOp);

		// Add the carry-in values.
		SegReduceSpine(preprocess.limitsDevice->get(), preprocess.numBlocks, 
			destDevice->get(), carryOutDevice->get(), identity, addOp,
			context);

		// Transform into the CSR space with BulkInsert.
		CsrBulkInsert(preprocess.csr2Device->get(), preprocess.numSegments, 
			destDevice->get(), identity, dest_global, context);

	} else {
		// No empties.

		// Evaluate the Spmv product.
		MGPU_MEM(T) carryOutDevice = context.Malloc<T>(preprocess.numBlocks);
		KernelSpmvApply<Tuning, LoadLeft>
			<<<preprocess.numBlocks, launch.x, 0, context.Stream()>>>(
			preprocess.threadCodesDevice->get(), matrix_global, cols_global,
			preprocess.count, vec_global, preprocess.limitsDevice->get(),
			dest_global, carryOutDevice->get(), identity, mulOp, addOp);
		MGPU_SYNC_CHECK("KernelSpmvApply");

		// Add the carry-in values.
		SegReduceSpine(preprocess.limitsDevice->get(), preprocess.numBlocks, 
			dest_global, carryOutDevice->get(), identity, addOp, context);
	}
}

template<typename ColsIt, typename VecIt, typename DestIt, typename T,
	typename MulOp, typename AddOp>
MGPU_HOST void SpmvUnaryApply(const SpmvPreprocessData& preprocess,
	ColsIt cols_global, VecIt vec_global, DestIt dest_global, T identity, 
	AddOp addOp, CudaContext& context) {

	SpmvApplyHost<false>(preprocess, (const T*)0, cols_global, vec_global,
		dest_global, identity, spmv_pass_through<T>(), addOp, context);
}

template<typename MatrixIt, typename ColsIt, typename VecIt, typename DestIt, 
	typename T, typename MulOp, typename AddOp>
MGPU_HOST void SpmvBinaryApply(const SpmvPreprocessData& preprocess,
	MatrixIt matrix_global, ColsIt cols_global, VecIt vec_global, 
	DestIt dest_global, T identity, MulOp mulOp, AddOp addOp,
	CudaContext& context) {

	SpmvApplyHost<true>(preprocess, matrix_global, cols_global, vec_global,
		dest_global, identity, mulOp, addOp, context);
}

} // namespace mgpu
