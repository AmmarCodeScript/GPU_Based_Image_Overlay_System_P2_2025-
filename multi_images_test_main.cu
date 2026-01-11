// main.cpp
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>

__global__ void overlayKernel(uchar4* base, uchar4* overlay, int width, int height) {
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

int main() {
    cv::Mat image = cv::imread("input.jpg", cv::IMREAD_UNCHANGED);
    cv::Mat logo  = cv::imread("Abo_Red.png", cv::IMREAD_UNCHANGED);

    if (image.empty() || logo.empty()) {
        std::cout << "ERROR: image or logo not found!" << std::endl;
        return -1;
    }

    if (image.channels() == 3) cv::cvtColor(image, image, cv::COLOR_BGR2BGRA);
    if (logo.channels() == 3)  cv::cvtColor(logo, logo, cv::COLOR_BGR2BGRA);

    cv::resize(logo, logo, cv::Size(200,200));

    // Skapa overlay-mat (text + logo)
    cv::Mat overlay(image.size(), CV_8UC4, cv::Scalar(0,0,0,0));
    logo.copyTo(overlay(cv::Rect(20,20,logo.cols,logo.rows)));

    cv::putText(overlay, "GPU Image Overlay",
                cv::Point(50, image.rows-50),
                cv::FONT_HERSHEY_SIMPLEX, 1.0,
                cv::Scalar(255,255,255,255), 2);
    cv::putText(overlay, "Date: 2025-01-01",
                cv::Point(50, image.rows-20),
                cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(255,255,255,255), 2);

    uchar4 *d_img, *d_overlay;
    size_t size = image.cols * image.rows * sizeof(uchar4);

    cudaMalloc(&d_img, size);
    cudaMalloc(&d_overlay, size);

    cudaMemcpy(d_img, image.ptr(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_overlay, overlay.ptr(), size, cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid((image.cols+15)/16, (image.rows+15)/16);

    auto start = std::chrono::high_resolution_clock::now();
    overlayKernel<<<grid, block>>>(d_img, d_overlay, image.cols, image.rows);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(image.ptr(), d_img, size, cudaMemcpyDeviceToHost);

    std::cout << "GPU Time: "
              << std::chrono::duration<double, std::milli>(end-start).count()
              << " ms\n";

    cv::imwrite("output_gpu.png", image);

    cudaFree(d_img);
    cudaFree(d_overlay);

    return 0;
}
