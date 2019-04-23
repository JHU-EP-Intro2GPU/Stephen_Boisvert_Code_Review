
// Public domain, header-only, OS-independent image I/O
// Thanks to https://github.com/nothings/stb for being brilliant
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Included in EN605.617.SPR19 examples
#include "helper_functions.h"
#include "helper_cuda.h"

// For existing cuda implementation comparison
#include <npp.h>

// I/O file names for test
const std::string output_image_name = "./test_edge_out.png";
const std::string nppi_output_image_name = "./test_nppi_edge_out.png";

// Helper for STB functions
__host__
void checkStbError(uint64_t ret_val, std::string desc) {
	if(0 == ret_val) {
		printf("STBI failed at step %s!\nExiting!\n", desc.c_str());
	}
}

// Constant values the gaussian filter
// Decision was made here to use a single-pass, two-dimensional filter over a two-pass filter
// The tradeoff is that we do more calculations, but fewer memory accesses, 
// which matches well with strengths of GPU
constexpr uint32_t GAUSSIAN_KERNEL_SIZE = 25;
constexpr uint32_t GAUSSIAN_KERNEL_MID = 12;
constexpr uint32_t GAUSSIAN_KERNEL_DIM = 5;
constexpr uint32_t GAUSSIAN_KERNEL_OFF = 3;


// sigma 6
/*
const float host_gaussian_vals[GAUSSIAN_KERNEL_SIZE] = {
    0.00947,    0.010435,   0.011184,	0.011658,	0.011821,	0.011658,	0.011184,	0.010435,	0.00947,
    0.010435,	0.011498,	0.012323,	0.012846,	0.013025,	0.012846,	0.012323,	0.011498,	0.010435,
    0.011184,	0.012323,	0.013207,	0.013767,	0.01396,	0.013767,	0.013207,	0.012323,	0.011184,
    0.011658,	0.012846,	0.013767,	0.014352,	0.014552,	0.014352,	0.013767,	0.012846,	0.011658,
    0.011821,	0.013025,	0.01396,	0.014552,	0.014755,	0.014552,	0.01396,	0.013025,	0.011821,
    0.011658,	0.012846,	0.013767,	0.014352,	0.014552,	0.014352,	0.013767,	0.012846,	0.011658,
    0.011184,	0.012323,	0.013207,	0.013767,	0.01396,	0.013767,	0.013207,	0.012323,	0.011184,
    0.010435,	0.011498,	0.012323,	0.012846,	0.013025,	0.012846,	0.012323,	0.011498,	0.010435,
    0.00947,	0.010435,	0.011184,	0.011658,	0.011821,	0.011658,	0.011184,	0.010435,	0.00947,
};
*/


// Sigma 2
const float host_gaussian_vals[GAUSSIAN_KERNEL_SIZE] = {
        0.023528, 0.033969, 0.038393, 0.033969, 0.023528,
        0.033969, 0.049045, 0.055432, 0.049045, 0.033969,
        0.038393, 0.055432, 0.062651, 0.055432, 0.038393,
        0.033969, 0.049045, 0.055432, 0.049045, 0.033969,
        0.023528, 0.033969, 0.038393, 0.033969, 0.023528
};

/*
// Sigma 1
const float host_gaussian_vals[GAUSSIAN_KERNEL_SIZE] = {
    0.003765, 0.015019, 0.023792, 0.015019, 0.003765,
    0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
    0.023792, 0.094907, 0.150342, 0.094907, 0.023792,
    0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
    0.003765, 0.015019, 0.023792, 0.015019, 0.003765
};
*/
__constant__ static float gpu_gaussian_vals[GAUSSIAN_KERNEL_SIZE];

// Sobel filters used for x and y axis intensity gradients
constexpr uint32_t SOBEL_KERNEL_SIZE = 9;
const int16_t host_sobel_x_vals[SOBEL_KERNEL_SIZE] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
};

const int16_t host_sobel_y_vals[SOBEL_KERNEL_SIZE] = {
    -1, -2, -1,
     0,  0,  0,
     1,  2,  1
};

// Constant memory buffers
__constant__ static uint16_t gpu_sobel_x_vals[SOBEL_KERNEL_SIZE];
__constant__ static uint16_t gpu_sobel_y_vals[SOBEL_KERNEL_SIZE];

// For converting radians to degrees - TODO: remove?
constexpr float CUSTOM_PI = 3.141592654;

///
/// \brief Create and syncronize a cuda event for timing
///

__host__ 
cudaEvent_t get_time(void)
{
    cudaEvent_t time;
    cudaDeviceSynchronize();
	cudaEventCreate(&time);
    cudaEventRecord(time);
    cudaEventSynchronize(time);
	return time;
}

///
/// \brief CUDA kernel which applies gaussian filter to image data
///
/// \param[in] img_data The image data to filter
/// \param[in] x_dim The x dimension of the image
/// \param[in] y_dim The y dimension of the image
/// \param[out] output Place to copy the smoothed image data
///

/// TODO: optimize
__global__
void customGaussianFilter(uint8_t* img_data, uint32_t x_dim, uint32_t y_dim, uint8_t* output) {

    // Find thread dimensions - define as equal to pixel co-ordinates
    uint32_t tid_x = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint32_t tid_y = (blockIdx.y * blockDim.y) + threadIdx.y;

    // Bounds checking
    if((tid_x < x_dim) && (tid_y < y_dim)) {

        int32_t total_pixels = x_dim * y_dim;
        int32_t pixel_idx = (x_dim * tid_y) + tid_x;
        float final_val = 0;
        for(int y = 1; y <= GAUSSIAN_KERNEL_DIM; y++) {
            for(int x = 1; x <= GAUSSIAN_KERNEL_DIM; x++) {
                int y_off = y - GAUSSIAN_KERNEL_OFF;
                int x_off = x - GAUSSIAN_KERNEL_OFF;
                int idx_off = (x_dim * y_off) + x_off;
                int gaus_off = (GAUSSIAN_KERNEL_DIM * y_off) + x_off;
                uint8_t image_val = ((pixel_idx + idx_off >= 0) && (pixel_idx + idx_off < total_pixels)) ?
                                    img_data[pixel_idx + idx_off] : img_data[pixel_idx];
                final_val += image_val * gpu_gaussian_vals[GAUSSIAN_KERNEL_MID + gaus_off];
            }
        }
        output[pixel_idx] = (uint8_t)final_val;

    }
}

///
/// \brief Applies a sobel filter for an intensity gradient approximation in both dimensions
/// 
/// \param[in] img_data The grayscale image data to filter
/// \param[in] x_dim The x dimension of the image data
/// \param[in] y_dim The y dimension of the image data
/// \param[out] x_output The place to copy the x-axis intensity gradient data
/// \param[out] y_output The place to copy the y-axis intensity gradient data
///

__global__
void customSobelFilter(uint8_t* img_data, uint32_t x_dim, uint32_t y_dim, int16_t* x_output, int16_t* y_output) {

    // Find thread dimensions - define as equal to pixel co-ordinates
    uint32_t tid_x = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint32_t tid_y = (blockIdx.y * blockDim.y) + threadIdx.y;

    // Bounds checking
    if((tid_x < x_dim) && (tid_y < y_dim)) {

        // It's a pretty small kernel - just explicitly do it
        bool left_edge = tid_x == 0;
        bool top_edge = tid_y == 0;
        bool right_edge = tid_x == (x_dim - 1);
        bool bottom_edge = tid_y == (y_dim - 1);

        uint32_t pixel_idx = (x_dim * tid_y) + tid_x;
        uint32_t pixel_idx_L = left_edge ? pixel_idx : pixel_idx - 1;
        uint32_t pixel_idx_R = right_edge ? pixel_idx : pixel_idx + 1;

        uint32_t pixel_idx_T = top_edge ? pixel_idx : pixel_idx - x_dim;
        uint32_t pixel_idx_TL = left_edge ? pixel_idx_T : pixel_idx_T - 1;
        uint32_t pixel_idx_TR = right_edge ? pixel_idx_T : pixel_idx_T + 1;

        uint32_t pixel_idx_B = bottom_edge ? pixel_idx : pixel_idx + x_dim;
        uint32_t pixel_idx_BL = left_edge ? pixel_idx_B : pixel_idx_B - 1;
        uint32_t pixel_idx_BR = right_edge ? pixel_idx_B : pixel_idx_B + 1;
        
        // Convolve with the sobel filters - can be more than 8 bits or negative sums
        int16_t x_total = 0;
        int16_t y_total = 0;

        // TODO: put pixel indices into an array and put this in a loop
        x_total += (int16_t)img_data[pixel_idx_TL] * gpu_sobel_x_vals[8];
        y_total += (int16_t)img_data[pixel_idx_TL] * gpu_sobel_y_vals[8];

        x_total += (int16_t)img_data[pixel_idx_T] * gpu_sobel_x_vals[7];
        y_total += (int16_t)img_data[pixel_idx_T] * gpu_sobel_y_vals[7];

        x_total += (int16_t)img_data[pixel_idx_TR] * gpu_sobel_x_vals[6];
        y_total += (int16_t)img_data[pixel_idx_TR] * gpu_sobel_y_vals[6];

        x_total += (int16_t)img_data[pixel_idx_L] * gpu_sobel_x_vals[5];
        y_total += (int16_t)img_data[pixel_idx_L] * gpu_sobel_y_vals[5];

        x_total += (int16_t)img_data[pixel_idx] * gpu_sobel_x_vals[4];
        y_total += (int16_t)img_data[pixel_idx] * gpu_sobel_y_vals[4];

        x_total += (int16_t)img_data[pixel_idx_R] * gpu_sobel_x_vals[3];
        y_total += (int16_t)img_data[pixel_idx_R] * gpu_sobel_y_vals[3];

        x_total += (int16_t)img_data[pixel_idx_BL] * gpu_sobel_x_vals[2];
        y_total += (int16_t)img_data[pixel_idx_BL] * gpu_sobel_y_vals[2];

        x_total += (int16_t)img_data[pixel_idx_B] * gpu_sobel_x_vals[1];
        y_total += (int16_t)img_data[pixel_idx_B] * gpu_sobel_y_vals[1];

        x_total += (int16_t)img_data[pixel_idx_BR] * gpu_sobel_x_vals[0];
        y_total += (int16_t)img_data[pixel_idx_BR] * gpu_sobel_y_vals[0];

        x_output[pixel_idx] = x_total;
        y_output[pixel_idx] = y_total;
        
    }

}

///
/// \brief Calculates intensity of a pixel gradient given the x and y axis components
///
/// \param[in] x_comp X component of the gradient
/// \param[in] y_comp Y component of the gradient
///
/// \return Total intensity of the gradient
///
__device__
float customGradientIntensity(float x_comp, float y_comp) {
    return sqrtf((x_comp * x_comp) + (y_comp * y_comp));
}

///
/// \brief Takes intensity gradient data and sets all non-local-maximum pixels to 0
///
/// \param[in] x_gradient_data Magnitudes of intensity gradient in x dimension
/// \param[in] y_gradient_data Magnitudes of intensity gradient in y dimension
/// \param[in] x_dim Number of pixels in X dimension of image
/// \param[in] y_dim Number of pixels in Y dimension of image
/// \param[out] output Place to copy output data
///
__global__
void customNonMaxSuppression(int16_t* x_gradient_data, int16_t* y_gradient_data, uint32_t x_dim, uint32_t y_dim, uint8_t* output) {
    
    // Find thread dimensions - define as equal to pixel co-ordinates
    uint32_t tid_x = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint32_t tid_y = (blockIdx.y * blockDim.y) + threadIdx.y;

    // Bounds checking
    if((tid_x < x_dim) && (tid_y < y_dim)) {
        
        // Determine direction of gradient
        uint32_t pixel_idx = (x_dim * tid_y) + tid_x;
        float arctan_radians = atan2f(y_gradient_data[pixel_idx], x_gradient_data[pixel_idx]);
        float arctan_degrees = ((arctan_radians > 0) ? arctan_radians : (2 * CUSTOM_PI + arctan_radians)) * 360 / (2 * CUSTOM_PI);

        bool left_edge = tid_x == 0;
        bool top_edge = tid_y == 0;
        bool right_edge = tid_x == (x_dim - 1);
        bool bottom_edge = tid_y == (y_dim - 1);
        uint32_t pixel_idx_L = left_edge ? pixel_idx : pixel_idx - 1;
        uint32_t pixel_idx_R = right_edge ? pixel_idx : pixel_idx + 1;

        uint32_t pixel_idx_T = top_edge ? pixel_idx : pixel_idx - x_dim;
        uint32_t pixel_idx_TL = left_edge ? pixel_idx_T : pixel_idx_T - 1;
        uint32_t pixel_idx_TR = right_edge ? pixel_idx_T : pixel_idx_T + 1;

        uint32_t pixel_idx_B = bottom_edge ? pixel_idx : pixel_idx + x_dim;
        uint32_t pixel_idx_BL = left_edge ? pixel_idx_B : pixel_idx_B - 1;
        uint32_t pixel_idx_BR = right_edge ? pixel_idx_B : pixel_idx_B + 1;

        bool vertical = ((arctan_degrees <= 22.5) || (arctan_degrees >= 337.5)) || ((arctan_degrees <= 202.5) && (arctan_degrees >= 157.5));
        bool horizontal = ((arctan_degrees >= 67.5) && (arctan_degrees <= 112.5)) || ((arctan_degrees >= 247.5) && (arctan_degrees <= 292.5));
        bool diag_down = ((arctan_degrees >= 292.5) && (arctan_degrees <=337.5)) || ((arctan_degrees >= 112.5) && (arctan_degrees <= 157.5));
        bool diag_up = ((arctan_degrees >= 202.5) && (arctan_degrees <= 247.5)) || ((arctan_degrees >= 22.5) && (arctan_degrees <= 67.5));

        // Compare magnitudes of pixel intensity in perpendicular direction to determine if it's a local max
        uint8_t pixel_intensity = customGradientIntensity(x_gradient_data[pixel_idx], y_gradient_data[pixel_idx]);
        bool local_max = false;
        if(vertical) {
            uint8_t pixel_intensity_L = customGradientIntensity(x_gradient_data[pixel_idx_L], y_gradient_data[pixel_idx_L]);
            uint8_t pixel_intensity_R = customGradientIntensity(x_gradient_data[pixel_idx_R], y_gradient_data[pixel_idx_R]);
            local_max = ((pixel_intensity >= pixel_intensity_R) && (pixel_intensity >= pixel_intensity_L));
        }
        else if(horizontal) {
            uint8_t pixel_intensity_T = customGradientIntensity(x_gradient_data[pixel_idx_T], y_gradient_data[pixel_idx_T]);
            uint8_t pixel_intensity_B = customGradientIntensity(x_gradient_data[pixel_idx_B], y_gradient_data[pixel_idx_B]);
            local_max = ((pixel_intensity >= pixel_intensity_T) && (pixel_intensity >= pixel_intensity_B));
        }
        else if(diag_down) {
            uint8_t pixel_intensity_BL = customGradientIntensity(x_gradient_data[pixel_idx_BL], y_gradient_data[pixel_idx_BL]);
            uint8_t pixel_intensity_TR = customGradientIntensity(x_gradient_data[pixel_idx_TR], y_gradient_data[pixel_idx_TR]);
            local_max = ((pixel_intensity >= pixel_intensity_TR) && (pixel_intensity >= pixel_intensity_BL));
        }
        else if(diag_up) {
            uint8_t pixel_intensity_TL = customGradientIntensity(x_gradient_data[pixel_idx_TL], y_gradient_data[pixel_idx_TL]);
            uint8_t pixel_intensity_BR = customGradientIntensity(x_gradient_data[pixel_idx_BR], y_gradient_data[pixel_idx_BR]);
            local_max = ((pixel_intensity >= pixel_intensity_BR) && (pixel_intensity >= pixel_intensity_TL));
        }
        output[pixel_idx] = local_max ? pixel_intensity : 0;
    }
}

///
/// \brief Applies canny edge detection to an input image
///
/// \param[in] img_data Grayscale image data to apply edge detection to
/// \param[in] x_dim X dimension of the image
/// \param[in] y_dim Y dimension of the image
/// \param[in] output_data Place to copy the output image
///

__host__
void customEdgeDetection(uint8_t* img_data, uint32_t x_dim, uint32_t y_dim) {


    // Determine number of blocks for the operation
    const dim3 BLOCK_DIM(16, 16);
    uint32_t data_size = x_dim * y_dim;
    dim3 num_blocks(x_dim/BLOCK_DIM.x, y_dim/BLOCK_DIM.y);
    if(x_dim % BLOCK_DIM.x != 0) {
        num_blocks.x += 1;
    }
    if(y_dim % BLOCK_DIM.y != 0) {
        num_blocks.y += 1;
    }

    // Create gpu buffers for the image
    uint8_t* gpu_img_data = nullptr;
    uint8_t* gpu_smoothed_img_data = nullptr;
    uint8_t* gpu_local_max_img_data = nullptr;
    checkCudaErrors(cudaMalloc(&gpu_img_data, data_size));
    checkCudaErrors(cudaMalloc(&gpu_smoothed_img_data, data_size));
    checkCudaErrors(cudaMalloc(&gpu_local_max_img_data, data_size));

    // Gradient values can be negative and > 256
    int16_t* gpu_x_gradient_img_data = nullptr;
    int16_t* gpu_y_gradient_img_data = nullptr;
    uint32_t sobel_data_size = data_size * sizeof(int16_t);
    checkCudaErrors(cudaMalloc(&gpu_x_gradient_img_data, sobel_data_size));
    checkCudaErrors(cudaMalloc(&gpu_y_gradient_img_data, sobel_data_size));


    // Start total timer
    cudaEvent_t startTime = get_time();
    
    // Prepare constant memory
    cudaMemcpyToSymbol(gpu_gaussian_vals, host_gaussian_vals, sizeof(host_gaussian_vals[0]) * GAUSSIAN_KERNEL_SIZE);
    cudaMemcpyToSymbol(gpu_sobel_x_vals, host_sobel_x_vals, sizeof(host_sobel_x_vals[0]) * SOBEL_KERNEL_SIZE);
    cudaMemcpyToSymbol(gpu_sobel_y_vals, host_sobel_y_vals, sizeof(host_sobel_y_vals[0]) * SOBEL_KERNEL_SIZE);
    
    // Copy input image to device
    checkCudaErrors(cudaMemcpy(gpu_img_data, img_data, data_size, cudaMemcpyHostToDevice));

    // Start edge timer
    cudaEvent_t startEdgeTime = get_time();

    // First apply gaussian filter
    customGaussianFilter<<<num_blocks, BLOCK_DIM>>>(gpu_img_data, x_dim, y_dim, gpu_smoothed_img_data);
  
    // Now find intensity gradient in x and y dimensions
    customSobelFilter<<<num_blocks, BLOCK_DIM>>>(gpu_smoothed_img_data, x_dim, y_dim, gpu_x_gradient_img_data, gpu_y_gradient_img_data);

    // Apply non-maximum suppression
    customNonMaxSuppression<<<num_blocks, BLOCK_DIM>>>(gpu_x_gradient_img_data, gpu_y_gradient_img_data, x_dim, y_dim, gpu_local_max_img_data);

	// Stop edge timer
    cudaEvent_t stopEdgeTime = get_time();

    // Copy output image to host
    checkCudaErrors(cudaMemcpy(img_data, gpu_local_max_img_data, data_size, cudaMemcpyDeviceToHost));

	// Stop overall 
    cudaEvent_t stopTime = get_time();
    
	// Report times
	float total_ms;
	float edge_ms;
	cudaEventElapsedTime(&total_ms, startTime, stopTime);
	cudaEventElapsedTime(&edge_ms, startEdgeTime, stopEdgeTime);
    printf("Finished edge in %f ms. %f total ms\n", edge_ms, total_ms);
    //
    // Used for outputting intensity gradients as a grayscale image for debug/inspection
    // TODO: put into nice debug function
    //

    /*
    int16_t* host_test_gradient_img_data = (int16_t*)malloc(sobel_data_size);
    checkCudaErrors(cudaMemcpy(host_test_gradient_img_data, gpu_y_gradient_img_data, sobel_data_size, cudaMemcpyDeviceToHost));
    for(uint32_t i = 0; i < data_size; i++) {
        img_data[i] = (host_test_gradient_img_data[i] / 4) + 128;
    }
    free(host_test_gradient_img_data);
    */

    // Free buffers
    cudaFree(gpu_img_data);
    cudaFree(gpu_smoothed_img_data);
    cudaFree(gpu_x_gradient_img_data);
    cudaFree(gpu_y_gradient_img_data);
    cudaEventDestroy(startTime);
    cudaEventDestroy(stopTime);
    cudaEventDestroy(startEdgeTime);
    cudaEventDestroy(stopEdgeTime);

}


///
/// \brief Executes an edge detection with npp library
///
/// \param[in] input_data The grayscale image data to use
/// \param[in] x_dim Width of the image
/// \param[in] y_dim Height of the image
/// \param[out] output_data Address to copy the edge detection of the input data
///

__host__
void runCudaEdgeCode(uint8_t* input_data, int x_dim, int y_dim, uint8_t* output_data) {

	// Data size for output - 1 byte per pixel
    size_t data_size = x_dim * y_dim;
    NppiSize img_size;
    img_size.height = y_dim;
    img_size.width = x_dim;

    // Scratch buffer size needed for edge detection
    int scratch_size;
    nppiFilterCannyBorderGetBufferSize(img_size, &scratch_size);
    printf("Scratch buffer size needed is %d\n", scratch_size);

	// Allocate cufftComplex input buffers for host and GPU
    Npp8u* gpu_input;
    Npp8u* gpu_output;
    Npp8u* gpu_scratch;
    checkCudaErrors(cudaMalloc(&gpu_input, data_size));
    checkCudaErrors(cudaMalloc(&gpu_output, data_size));
    checkCudaErrors(cudaMalloc(&gpu_scratch, scratch_size));

    // Start timer
    cudaEvent_t startTime = get_time();
    
    // Copy input image to device
    checkCudaErrors(cudaMemcpy(gpu_input, input_data, data_size, cudaMemcpyHostToDevice));

    // Start edge timer
    cudaEvent_t startEdgeTime = get_time();
    
    // Do edge detection
    //checkCudaErrors(cudaMemcpy(gpu_output, gpu_input, data_size, cudaMemcpyDeviceToDevice));
    NppiPoint offset;
    offset.x = 0;
    offset.y = 0;
    nppiFilterCannyBorder_8u_C1R(gpu_input, x_dim, img_size, offset, gpu_output, x_dim, img_size, NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3, 50, 150, nppiNormL1, NPP_BORDER_REPLICATE, gpu_scratch);

	// Stop edge timer
    cudaEvent_t stopEdgeTime = get_time();
    
    // Copy output image to host
    checkCudaErrors(cudaMemcpy(output_data, gpu_output, data_size, cudaMemcpyDeviceToHost));

	// Stop overall 
    cudaEvent_t stopTime = get_time();
    
	// Report times
	float total_ms;
	float edge_ms;
	cudaEventElapsedTime(&total_ms, startTime, stopTime);
	cudaEventElapsedTime(&edge_ms, startEdgeTime, stopEdgeTime);
    printf("Finished edge in %f ms. %f total ms\n", edge_ms, total_ms);
    
    // Free device buffers
    cudaFree(gpu_input);
    cudaFree(gpu_output);
    cudaFree(gpu_scratch);
    cudaEventDestroy(startTime);
    cudaEventDestroy(stopTime);
    cudaEventDestroy(startEdgeTime);
    cudaEventDestroy(stopEdgeTime);
}

__host__
int main(int argc, char** argv) {
    
	// Input is image name
	if(argc != 2) {
		printf("Execute with image name on commandline. e.g. ./edge_detection img.png\n");
		exit(1);
	}
    const std::string input_image_name(argv[1]);

    // Force grayscale
	int num_components = 1;
	int x_dim, y_dim, max_components;

	// Read input image
	uint8_t* stb_img_data = stbi_load(input_image_name.c_str(), &x_dim, &y_dim, 
										&max_components, num_components);
	checkStbError((uint64_t)stb_img_data, "Loading image " + input_image_name);
    printf("Read image x_dim=%d, y_dim=%d\n", x_dim, y_dim);

    // Copy data to keep original data
    uint8_t* stb_out_img_data = (uint8_t*)STBIW_MALLOC(x_dim * y_dim);
    memcpy(stb_out_img_data, stb_img_data, x_dim * y_dim);

    //
    // Do our custom edge detection
    //
    customEdgeDetection(stb_out_img_data, x_dim, y_dim);

    for(uint32_t i = 0; i < x_dim * y_dim; i++) {
        if(stb_out_img_data[i] > 175) {
            stb_out_img_data[i] = 255;
        }
        else if(stb_out_img_data[i] > 75) {
            stb_out_img_data[i] = 128;
        }
        else {
            stb_out_img_data[i] = 0;
        }
    }
    
	// Write the image back as output
	checkStbError(stbi_write_png(output_image_name.c_str(), x_dim, y_dim, 
									num_components, stb_out_img_data, x_dim),
									"Writing image " + output_image_name);
    printf("Wrote image %s!\n", output_image_name.c_str());
    
    // Run with npp library to compare times
    uint8_t* stb_nppi_img_data = (uint8_t*)STBIW_MALLOC(x_dim * y_dim);
    runCudaEdgeCode(stb_img_data, x_dim, y_dim, stb_nppi_img_data);
    checkStbError(stbi_write_png(nppi_output_image_name.c_str(), x_dim, y_dim, 
                                    num_components, stb_nppi_img_data, x_dim),
                                    "Writing image " + nppi_output_image_name);
    printf("Wrote image %s!\n", nppi_output_image_name.c_str());

    // Free stb buffers
    STBI_FREE(stb_img_data);
    STBIW_FREE(stb_out_img_data);
    STBIW_FREE(stb_nppi_img_data);
    return 0;
}