// overlay.cu
#include <cuda_runtime.h>

__global__ void overlayKernel(
    uchar4* base,
    uchar4* overlay,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        uchar4 bg = base[idx];
        uchar4 fg = overlay[idx];

        float alpha = fg.w / 255.0f;

        bg.x = (1 - alpha) * bg.x + alpha * fg.x;
        bg.y = (1 - alpha) * bg.y + alpha * fg.y;
        bg.z = (1 - alpha) * bg.z + alpha * fg.z;

        base[idx] = bg;
    }
}
