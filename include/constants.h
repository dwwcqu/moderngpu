#pragma once

#define MGPU_NT 64 //128
#define MGPU_VT 4  //8: Must be greater or equal to 4
#define MGPU_NV MGPU_NT*MGPU_VT
#define MGPU_TB 8  //4: Must be power of 4
#define MGPU_BC 64
