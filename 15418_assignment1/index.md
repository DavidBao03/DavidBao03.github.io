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

## project 3 -- ISPC
该任务的第一部分是让我们用ispc重写第一个任务。只需要编译运行即可。
![](/img/ispc_running1.png "ISPC Running Result")
可以看到，仅仅是使用ispc，就获得了5.84x的加速比。

第二部分是让我们使用ispc task，这样可以将任务分发在更多的cpu核上，而不仅仅是在单个核上进行SIMD操作。
![](/img/ispc_running2.png "ISPC Tasks Running Result")
使用了task之后，加速比得到了更大的提升，从5.84x提升到了11.69x。

为了进一步提升性能，我们尝试修改task的数量，并得到了以下的结果：

| Task数量 |   2    |   4    |   8    |   16   |   32   |
| :------: | :----: | :----: | :----: | :----: | :----: |
|  加速比  | 11.69x | 13.97x | 18.95x | 18.82x | 19.18x |

## project 4 -- Sqrt
该任务编写了一个计算2000万个在0到3之间随机数的平方根程序。我们以单核和多核（with or without task）的版本运行：
| 是否开启Task |   否    |   是    |
| :------: | :----: | :----: |
|  加速比  | 4.81x | 18.38x |

为了让加速比最大，我们需要尽可能的让每一次计算都更久一些。从handout上的图片来看，3附近的值需要迭代的次数最多。因为，我们可以将所有的值都改为2.998f：
```c++
for (unsigned int i=0; i<N; i++)
    {
        // TODO: CS149 students.  Attempt to change the values in the
        // array here to meet the instructions in the handout: we want
        // to you generate best and worse-case speedups
        
        // starter code populates array with random input values
        // values[i] = .001f + 2.998f * static_cast<float>(rand()) / RAND_MAX;
        //best case:
        values[i] = 2.998f;
    }
```
我们重新运行程序，可以得到以下的结果：
| 是否开启Task |   否    |   是    |
| :------: | :----: | :----: |
|  加速比  | 6.70x | 21.92x |

为了让加速比最小，我们可以让SIMD中每一个向量尽可能的“浪费”。具体来说，如果一个向量长为8，我们可以只让其中一个数据运行的尽量久，而其他的数据尽可能的“简单”（计算时间短），这样根据“木桶效应”，一个向量的计算时间被计算时间最长的数据所决定，导致了7个数据位置计算时间的浪费，具体地，我们可以这样分配我们的数组：
```c++
for (unsigned int i=0; i<N; i++)
    {
        // TODO: CS149 students.  Attempt to change the values in the
        // array here to meet the instructions in the handout: we want
        // to you generate best and worse-case speedups
        
        // starter code populates array with random input values
        // values[i] = .001f + 2.998f * static_cast<float>(rand()) / RAND_MAX;
        //best case:
        // values[i] = 2.998f;
        //worst case:
        if(i % 8 == 0) values[i] =2.998f;
        else values[i] = 1.f;
    }
```
我们重新运行程序，可以得到以下的结果：
| 是否开启Task |   否    |   是    |
| :------: | :----: | :----: |
|  加速比  | 0.93x | 3.03x |

## project 5 -- BLAS saxpy
该任务是saxpy的实现。saxpy指的是result = sacle * X + Y， 其中X，Y均为向量，而sacle是标量。

编译运行的结果如下：
![](/img/ispc_running3.png)

阅读代码后，得知使用了64个ispc task，但是加速比仅仅只有1.3x。观察到后面的吞吐量，可以推断出是吞吐量的瓶颈导致了程序难以进一步优化。

> Extra Credit: (1 point) Note that the total memory bandwidth consumed computation in main.cpp is TOTAL_BYTES = 4 * N * sizeof(float);. Even though saxpy loads one element from X, one element from Y, and writes one element to result the multiplier by 4 is correct. Why is this the case? (Hint, think about how CPU caches work.)

之所以乘以4的原因是，如果取X和Y的过程中缓存未命中，cpu会先将值拷贝至缓存，再从缓存中拷贝值至内存。

## project 6 -- K-means
疑似Stanford没有公开这里的资源，所以这个任务就摆烂了吧！
