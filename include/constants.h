#pragma once

#define MGPU_NTX 256
#define MGPU_NTY 1
#define MGPU_NTZ 1
#define MGPU_NT  MGPU_NTX*MGPU_NTY //64
#define MGPU_VT  1  //4: Must be greater or equal to 4
#define MGPU_NV  MGPU_NTX*MGPU_VT
#define MGPU_TB  4  //8: Must be power of 4
#define MGPU_BC  64
