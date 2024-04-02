# 15418_assignment1


# 15418 Assignment1
## project 1 -- Mandelbrot
作业中的第一个任务是描绘曼德拉分形，具体代码可以参考作业提供的*mandelbrotSerial*函数和*mandel*函数：
```c++
void mandelbrotSerial(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int startRow, int totalRows,
    int maxIterations,
    int output[])
{
    float dx = (x1 - x0) / width;
    float dy = (y1 - y0) / height;

    int endRow = startRow + totalRows;

    for (int j = startRow; j < endRow; j++) {
        for (int i = 0; i < width; ++i) {
            float x = x0 + i * dx;
            float y = y0 + j * dy;

            int index = (j * width + i);
            output[index] = mandel(x, y, maxIterations);
        }
    }
}
```

```c++
static inline int mandel(float c_re, float c_im, int count)
{
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i) {

        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re*z_re - z_im*z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}
```

这两个函数的具体含义不展开描述，但是关键是可以看出*Serial*函数需要提供*totalRows*参数，这意味着*Serial*函数是将整张图片一起写入的。第一个任务的目标就是要将*Serial*函数并行化。具体可以参考*mandelbrotThread*函数：
```c++
void mandelbrotThread(
    int numThreads,
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations, int output[])
{
    static constexpr int MAX_THREADS = 32;

    if (numThreads > MAX_THREADS)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", MAX_THREADS);
        exit(1);
    }

    // Creates thread objects that do not yet represent a thread.
    std::thread workers[MAX_THREADS];
    WorkerArgs args[MAX_THREADS];

    for (int i=0; i<numThreads; i++) {
      
        // TODO FOR CS149 STUDENTS: You may or may not wish to modify
        // the per-thread arguments here.  The code below copies the
        // same arguments for each thread
        args[i].x0 = x0;
        args[i].y0 = y0;
        args[i].x1 = x1;
        args[i].y1 = y1;
        args[i].width = width;
        args[i].height = height;
        args[i].maxIterations = maxIterations;
        args[i].numThreads = numThreads;
        args[i].output = output;
      
        args[i].threadId = i;

    }

    // Spawn the worker threads.  Note that only numThreads-1 std::threads
    // are created and the main application thread is used as a worker
    // as well.
    for (int i=1; i<numThreads; i++) {
        workers[i] = std::thread(workerThreadStart, &args[i]);
    }
    
    workerThreadStart(&args[0]);

    // join worker threads
    for (int i=1; i<numThreads; i++) {
        workers[i].join();
    }
}
```

具体来说，*thread*函数内部创建了*numThreads*个不同的线程，分别需要完成*workerThreadStart*工作：
```c++
void workerThreadStart(WorkerArgs * const args) {

    // TODO FOR CS149 STUDENTS: Implement the body of the worker
    // thread here. Each thread should make a call to mandelbrotSerial()
    // to compute a part of the output image.  For example, in a
    // program that uses two threads, thread 0 could compute the top
    // half of the image and thread 1 could compute the bottom half.

}
```

要将函数并行化，我们首先编写一个辅助函数，将上述的*totalRows*参数改为每一个线程所需要计算的图像：
```c++
void mandelbrotStepSerial(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int startRow, int step,
    int maxIterations,
    int output[])
{
    float dx = (x1 - x0) / width;
    float dy = (y1 - y0) / height;

    for (int j = startRow; j < height; j+=step) {
        for (int i = 0; i < width; ++i) {
            float x = x0 + i * dx;
            float y = y0 + j * dy;

            int index = (j * width + i);
            output[index] = mandel(x, y, maxIterations);
        }
    }
}
```

最后，我们在*workerThreadStart*函数里调用这个函数即可：
```c++
void workerThreadStart(WorkerArgs * const args) {

    // TODO FOR CS149 STUDENTS: Implement the body of the worker
    // thread here. Each thread should make a call to mandelbrotSerial()
    // to compute a part of the output image.  For example, in a
    // program that uses two threads, thread 0 could compute the top
    // half of the image and thread 1 could compute the bottom half.

    mandelbrotStepSerial(args->x0, args->y0, args->x1, args->y1,
    args->width, args->height, args->threadId, args->numThreads, args->maxIterations, args->output);
}
```

根据作业要求，我们需要测试线程数量不同时的加速比：
| 线程数量 |   2  |   3  |  4   |   5  |  6   |  7   |   8  |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 加速比  | 2.00x |   2.79x   |   3.72x   | 3.17x | 3.42x | 3.59x | 3.64x |

可以看出，随着线程数量的增加，在一开始加速比有明显的提升。但是当线程到达一定数量后（这里是3个线程），加速比不再继续增加了。

## project 2 -- SIMD
第二个任务是让我们使用SIMD进行编程。不同于直接操作汇编指令，作业提供了一个“伪SIMD”库使用。我们的任务是用SIMD编写*clampedExpVector*函数。该函数的串行版本为*clampedExpSerial*。由于任务比较简单，这里直接给出代码：
```c++
void clampedExpVector(float* values, int* exponents, float* output, int N) {

  //
  // CS149 STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  int i;
  __cs149_vec_float x;
  __cs149_vec_float result;
  __cs149_vec_int exp;
  __cs149_vec_int ones = _cs149_vset_int(1);
  __cs149_vec_int zero = _cs149_vset_int(0);
  __cs149_vec_float clamp = _cs149_vset_float(9.999999f);
  __cs149_mask maskAll = _cs149_init_ones();
  __cs149_mask maskLess, maskClamp, maskNotClamp;
  for (i = 0; i + VECTOR_WIDTH <= N; i += VECTOR_WIDTH) {
    result = _cs149_vset_float(1.f);

    _cs149_vload_int(exp, exponents+i, maskAll);
    _cs149_vload_float(x, values+i, maskAll);
    _cs149_vgt_int(maskNotClamp, exp, zero, maskAll);
    
    while(_cs149_cntbits(maskNotClamp)) {
      _cs149_vmult_float(result, result, x, maskNotClamp);
      _cs149_vsub_int(exp, exp, ones, maskNotClamp);
      _cs149_vgt_int(maskNotClamp, exp, zero, maskAll);
    }

    _cs149_vgt_float(maskLess, result, clamp, maskAll);
    _cs149_vset_float(result, 9.999999f, maskLess);

    // Write results back to memory
    _cs149_vstore_float(output+i, result, maskAll);
  }

  i -= VECTOR_WIDTH;
  if (i != N) {
    clampedExpSerial(values + i, exponents + i, output + i, N - i);
  }
}
```

针对不同的向量长度，有以下测试数据：
|  向量长度  |   2   |   4   |   8   |  16   |
| :--------: | :---: | :---: | :---: | :---: |
| 向量利用率 | 90.6% | 86.4% | 81.0% | 81.6% |

可以简单地认为，向量长度越长，向量的利用率越差。我个人的理解是：向量长度越长，越可能导致数据无法填充整个向量，并且如果数据长度不是向量长度的倍数，最后余下的数据只能进行常规的串行计算。

Extra：完成*arraySumVector*函数：
```c++
float arraySumVector(float* values, int N) {
  
  //
  // CS149 STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  float ans[VECTOR_WIDTH] = {0};
  __cs149_vec_float val;
  __cs149_vec_float res = _cs149_vset_float(0.f);
  __cs149_mask maskAll = _cs149_init_ones();
  for (int i=0; i < N; i+=VECTOR_WIDTH) {
    _cs149_vload_float(val, values + i, maskAll);
    _cs149_vadd_float(res, res, val, maskAll);
  }
  
  int i = VECTOR_WIDTH;
  while (i /= 2) {
    _cs149_hadd_float(res, res);
    _cs149_interleave_float(res, res);
  }

  _cs149_vstore_float(ans, res, maskAll);

  return ans[0];
}
```


