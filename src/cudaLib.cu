
#include "cudaLib.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
        if (code != cudaSuccess)
        {
                fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
                if (abort) exit(code);
        }
}

__global__
void saxpy_gpu (float* x, float* y, float scale, int size) {
        //      Insert GPU SAXPY kernel code here
        int i = blockDim.x*blockIdx.x + threadIdx.x;
        if(i<size) y[i] = scale*x[i] + y[i];
}

int runGpuSaxpy(int vectorSize) {

        //std::cout << "Hello GPU Saxpy!\n";

        float *h_x, *h_y, *h_o;
        float *d_x, *d_y;
        int size = vectorSize * sizeof(float);
        int no_of_blks = (vectorSize + 1023)/1024;
        int scale = (float)(rand() % 100);
        int no_of_errors = 0;

        h_x = (float *)malloc(size);
        h_y = (float *)malloc(size);
        h_o = (float *)malloc(size);

        vectorInit(h_x,vectorSize);
        vectorInit(h_y,vectorSize);

        cudaMalloc((void **)&d_x,size);
        cudaMalloc((void **)&d_y,size);

        cudaMemcpy(d_x,h_x,size,cudaMemcpyHostToDevice);
        cudaMemcpy(d_y,h_y,size,cudaMemcpyHostToDevice);
        saxpy_gpu<<<no_of_blks,1024>>>(d_x,d_y,scale,vectorSize);

        cudaDeviceSynchronize();
        cudaMemcpy(h_o,d_y,size,cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        cudaFree(d_x);
        cudaFree(d_y);

        //h_o[0] = 0;

        no_of_errors = verifyVector(h_x,h_y,h_o,scale,vectorSize);

        printf("No of errors : %d / %d",no_of_errors,vectorSize);

        //      Insert code here
        //std::cout << "Lazy, you are!\n";
        //std::cout << "Write code, you must\n";

        return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
        //      Insert code here
        curandState_t rng;
        int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
        curand_init(clock64(), idx, 0, &rng);

        if(idx < pSumSize){
                for(int i = 0;i<sampleSize;i++){
                        float x = curand_uniform(&rng);
                        float y = curand_uniform(&rng);
                        if((x*x + y*y) <= 1.0f){
                                pSums[idx]++;
                        }
                }
        }
}

__global__
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
        //      Insert code here
        int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
        int reduce_thread_count = pSumSize / reduceSize;
        int lower_limit;
        if(idx < reduce_thread_count){
                lower_limit = idx * reduceSize;
                totals[idx] = 0;
                for(int i=lower_limit; i<(lower_limit+reduceSize); i++){
                        totals[idx] += pSums[i];
                }
        }
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize,
        uint64_t reduceThreadCount, uint64_t reduceSize) {

        //  Check CUDA device presence
        int numDev;
        cudaGetDeviceCount(&numDev);
        if (numDev < 1) {
                std::cout << "CUDA device missing!\n";
                return -1;
        }
        auto tStart = std::chrono::high_resolution_clock::now();

        float approxPi = estimatePi(generateThreadCount, sampleSize,
                reduceThreadCount, reduceSize);

        std::cout << "Estimated Pi = " << approxPi << "\n";

        auto tEnd= std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> time_span = (tEnd- tStart);
        std::cout << "It took " << time_span.count() << " seconds.";

        return 0;
}


double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize,
        uint64_t reduceThreadCount, uint64_t reduceSize) {

        double approxPi = 0;
        uint64_t final_count = 0;
        uint64_t *h_totals, *d_pSums, *d_totals;
        int no_of_blks_generatePoints = (generateThreadCount + 1023)/1024;

        int no_of_blks_reduceCounts = (reduceThreadCount + 1023)/1024;

        //      Insert code here
        //std::cout << "Sneaky, you are ...\n";
        //std::cout << "Compute pi, you must!\n";

        long long int total_sample_size = generateThreadCount * sampleSize;
        //int total_thread_count = generateThreadCount * 256;

        h_totals = (uint64_t *)malloc(reduceThreadCount * sizeof(uint64_t));

        cudaMalloc((void **)&d_pSums,generateThreadCount * sizeof(uint64_t));

        cudaMalloc((void **)&d_totals,reduceThreadCount * sizeof(uint64_t));

        generatePoints<<<no_of_blks_generatePoints, 1024>>>(d_pSums,generateThreadCount,sampleSize);
        cudaDeviceSynchronize();

        //cudaMalloc((void **)&reduced_partial_sum_array,(reduceSize * sizeof(int)));
        reduceCounts<<<no_of_blks_reduceCounts, 1024>>>(d_pSums,d_totals,generateThreadCount,reduceSize);

        cudaDeviceSynchronize();

        cudaMemcpy(h_totals,d_totals,reduceThreadCount * sizeof(uint64_t),cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        cudaFree(d_totals);

        cudaFree(d_pSums);

        for(int i=0; i<reduceThreadCount; i++)
        {
                final_count += h_totals[i];
        }

        approxPi = (((float)(final_count))/total_sample_size) * 4;

        printf("final_count : %lld", final_count);
        printf("Sample_size_per_thread : %d",sampleSize);
        printf("Total Sample size : %lld", total_sample_size);
        printf("Approximate value of pi from MonteCarlo simulation is %lf", approxPi);

        return approxPi;
}
