// https://github.com/grensen/cnn24
// .NET 8
// copy code
// then press to close functions: (ctrl + m + o)
// use Release mode, Debug is slow!

// auto download and extract cifar10 dataset from:
// https://www.cs.toronto.edu/~kriz/cifar.html
// copy to string path = @"C:\cifar10\";

using System.Diagnostics;
using System.IO.Compression;
using System.Net;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Text;

class cnn24_cifar10
{
    static void Main(string[] args)
    {
        // 0. load cifar10 data
        Console.WriteLine("\nBegin cnn24 cifar10 overfitting demo\n");

        string path = @"C:\cifar10\";
        AutoCifar10 d = new AutoCifar10(path);

        // 1. init cnn + nn + hyperparameters
        float scaler = 4.0f;
        int[] cnn = { 3, (int)(scaler * 6), (int)(scaler * 12) }; // input layer: non-RGB = 1, RGB = 3
        int[] filter = { 3, 3 }; // x * y dim for kernel
        int[] stride = { 2, 2 }; // "pooling" with higher strides than 1
        int[] net = { 3072, (int)(1 * 800), (int)(1 * 800), 10 }; // nn

        var LR = 0.003f;
        var MOMENTUM = 0.3f;
        var DROPOUT = 0.0f;
        var SEED = 274024;
        var FACTOR = 0.93f;
        int BATCH = 50;
        int EPOCHS = 50;
        bool shuffle = true;
        bool flip = false;
        bool autoFilter = false;
        float randomFilter = 0.0f;
        int shift = 0;
        float L2 = 0.1f;

        // 2.0 convolution dimensions
        int[] dim = CnnDimensions(cnn.Length - 1, 32, filter, stride);
        // 2.1 convolution steps for layerwise preparation
        int[] cStep = CnnSteps(cnn, dim);
        // 2.2 kernel steps for layerwise preparation
        int[] kStep = KernelSteps(cnn, filter);
        // 2.3 init visual based kernel weights
        float[] kernel = InitConvKernel(cnn, filter, SEED);
        // 2.4 init neural network weights w.r.t. cnn
        float[] weights = InitNeuralNetWeights(net, cStep, SEED);

        // hyper parameter backup 
        float lr = LR, mom = MOMENTUM, l2 = L2;
        NetInfo();

        Console.WriteLine("\nStart training");
        // overfitting
        RunTraining();

        // auto filter
        autoFilter = true;
        Console.WriteLine($"Activate: AUTOFILTER = {autoFilter}");
        RunTraining();

        // flip, random filter, shift
        randomFilter = 0.2f;
        flip = true;
        shift = 3;
        Console.WriteLine($"Activate: XFLIP = {flip:F0}, RANDFILTER = {randomFilter:F2}, SHIFT = {shift:F0}");

        RunTraining();

        Console.WriteLine("End CNN demo");

        void NetInfo()
        {
#if DEBUG
                Console.WriteLine("Debug mode is on, switch to Release mode"); return;
#endif

            Console.WriteLine($"Convolution = {string.Join("-", cnn)}");
            Console.WriteLine($"Kernel size = {string.Join("-", filter)}");
            Console.WriteLine($"Stride step = {string.Join("-", stride)}");
            Console.WriteLine($"DimensionX{" ",2}= {string.Join("-", dim)}");
            Console.WriteLine($"Map (Dim²){" ",2}= {string.Join("-", dim.Select(x => x * x))}");
            Console.WriteLine($"CNN+NN{" ",6}=" +
                $" {string.Join("-", cnn.Zip(dim, (x, d) => x * d * d))}+{string.Join("-", net)}");
            Console.WriteLine($"CNN+NN WTS{" ",2}= ({kernel.Length} ({cnn.Zip(cnn.Skip(1),
            (p, n) => p * n).Sum()}))" + $"+({weights.Length})");
            Console.WriteLine($"SEED{" ",8}= {SEED:F0}");
            Console.WriteLine($"EPOCHS{" ",6}= {EPOCHS:F0}");
            Console.WriteLine($"BATCHSIZE{" ",3}= {BATCH:F0}");
            Console.WriteLine($"CNN LR{" ",6}= {LR:F4} | MLT = {FACTOR:F2}");
            Console.WriteLine($"NN LR{" ",7}= {LR:F4} | MLT = {FACTOR:F2}");
            Console.WriteLine($"MOMENTUM{" ",4}= {MOMENTUM:F2}{" ",3}| MLT = {FACTOR:F2}");
            if (L2 != 0) Console.WriteLine($"L2NORM{" ",6}= {L2:F2}{" ",3}"
                + $"| MLT = {FACTOR:F2}"
                );
            if (DROPOUT != 0) Console.WriteLine($"DROPOUT+{" ",4}= {DROPOUT:F2}");
            Console.WriteLine($"SHUFFLE{" ",5}= {shuffle:F0}");

        }

        void RunTraining()
        {
            // reset settings
            LR = lr; MOMENTUM = mom; L2 = l2;
            // init weights
            kernel = InitConvKernel(cnn, filter, SEED);
            weights = InitNeuralNetWeights(net, cStep, SEED);

            Random rng = new(SEED);
            Stopwatch sw = Stopwatch.StartNew();
            int[] indices = Enumerable.Range(0, d.cifarTrainbLabel.Length).ToArray(),
                done = new int[d.cifarTrainbLabel.Length];

            // 5.0
            float[] delta = new float[weights.Length], kDelta = new float[kernel.Length];
            for (int epoch = 0; epoch < EPOCHS; epoch++, LR *= FACTOR, MOMENTUM *= FACTOR, L2 *= FACTOR)
            {
                CnnTraining(true, indices, d.cifarTrainNorm, d.cifarTrainbLabel, d.cifarTestNorm, d.cifarTestLabel, rng.Next(), done, cnn, filter, stride,
                     kernel, kDelta, dim, cStep, kStep, net, weights, delta,
                     epoch, BATCH, LR, L2, MOMENTUM, DROPOUT, shuffle, flip, shift, autoFilter, randomFilter);
            }
            Console.WriteLine($"Done after {(sw.Elapsed.TotalMilliseconds / 1000.0):F2}s\n");
            // CnnTesting(d.cifarTestNorm, d.cifarTestLabel, cnn, filter, stride, kernel, dim, cStep, kStep, net, weights, 10000, autoFilter);
        }
    }

    //+-------------------------------------------------------------------------------------------+

    // 2.0 conv dimensions
    static int[] CnnDimensions(int cnn_layerLen, int startDimension, int[] filter, int[] stride)
    {
        int[] dim = new int[cnn_layerLen + 1];
        for (int i = 0, c_dim = (dim[0] = startDimension); i < cnn_layerLen; i++)
            dim[i + 1] = c_dim = (c_dim - (filter[i] - 1)) / stride[i];
        return dim;
    }
    // 2.1 convolution steps
    static int[] CnnSteps(int[] cnn, int[] dim)
    {
        int[] cs = new int[cnn.Length + 1];
        cs[1] = dim[0] * dim[0] * cnn[0]; // startDimension^2
        for (int i = 0, sum = cs[1]; i < cnn.Length - 1; i++)
            cs[i + 2] = sum += cnn[i + 1] * dim[i + 1] * dim[i + 1];
        return cs;
    }
    // 2.2 kernel steps in structure for kernel weights
    static int[] KernelSteps(int[] cnn, int[] filter)
    {
        int[] ks = new int[cnn.Length - 1];
        for (int i = 0; i < cnn.Length - 2; i++)
            ks[i + 1] += cnn[i + 0] * cnn[i + 1] * filter[i] * filter[i];
        return ks;
    }
    // 2.3 init kernel weights
    static float[] InitConvKernel(int[] cnn, int[] filter, int seed)
    {
        int cnn_weightLen = 0;
        for (int i = 0; i < cnn.Length - 1; i++)
            cnn_weightLen += cnn[i] * cnn[i + 1] * filter[i] * filter[i];
        float[] kernel = new float[cnn_weightLen]; Random rnd = new(seed);
        for (int i = 0, c = 0; i < cnn.Length - 1; i++) // each cnn layer
            for (int l = 0, f = filter[i]; l < cnn[i]; l++) // each input map
                for (int r = 0; r < cnn[i + 1]; r++) // each output map
                    for (int row = 0; row < f; row++) // kernel y
                        for (int col = 0; col < f; col++, c++) // kernel x
                            kernel[c] = //(rnd.NextSingle() * 2 - 1) /  MathF.Sqrt(cnn[i] * cnn[i + 1] * 0.5f);
                                 (rnd.NextSingle() * 2 - 1)
                * MathF.Sqrt(6.0f / (cnn[i] + cnn[i + 1])) * 0.5f;
        return kernel;
    }
    // 2.4 init neural network weights for cnn
    static float[] InitNeuralNetWeights(int[] net, int[] convMapsStep, int seed)
    {
        // 3.1.1 fit cnn output to nn input
        net[0] = convMapsStep[^1] - convMapsStep[^2];
        // 3.1.2 glorot nn weights init
        int len = 0;
        for (int n = 0; n < net.Length - 1; n++)
            len += net[n] * net[n + 1];
        float[] weight = new float[len];
        Random rnd = new(seed);
        for (int i = 0, m = 0; i < net.Length - 1; i++, m += net[i] * net[i - 1]) // layer
            for (int w = m; w < m + net[i] * net[i + 1]; w++) // weights
                weight[w] = (rnd.NextSingle() * 2 - 1)
                * MathF.Sqrt(6.0f / (net[i] + net[i + 1])) * 0.5f;
        return weight;
    }
    // 2.5 cnn number of neurons
    static int CnnNeuronsLen(int startDimension, int[] cnn, int[] dim)
    {
        int cnn_layerLen = cnn.Length - 1;
        int cnn_neuronLen = startDimension * startDimension * cnn[0]; // add input first
        for (int i = 0; i < cnn_layerLen; i++)
            cnn_neuronLen += cnn[i + 1] * dim[i + 1] * dim[i + 1];
        return cnn_neuronLen;
    }
    // 2.6 nn number of neurons
    static int NeuronsLen(int[] net)
    {
        int sum = 0;
        for (int n = 0; n < net.Length; n++)
            sum += net[n];
        return sum;
    }

    // 3.0.1 horizontal flip
    static void Flip(Span<float> sample)
    {
        int width = 32, height = 32;
        for (int y = 0; y < height; y++)
        {
            int start = y * width, end = (y + 1) * width - 1;
            while (start < end)
                (sample[start], sample[end]) = (sample[end--], sample[start++]);
        }
    }
    // 3.0.2 random filter
    static void RandomFilter(Span<float> image, Span<float> randomKernel, int width = 32)
    {
        Span<float> sharpenedImage = stackalloc float[image.Length];
        for (int y = 1; y < width - 1; y++)
            for (int x = 1; x < width - 1; x++)
            {
                float convolutedValue = 0;
                int kernelIndex = 0;
                for (int ky = -1; ky <= 1; ky++)
                    for (int kx = -1; kx <= 1; kx++)
                    {
                        int pixelX = x + kx, pixelY = y + ky;
                        convolutedValue += image[pixelY * width + pixelX] * randomKernel[kernelIndex++];
                    }
                sharpenedImage[y * width + x] = convolutedValue;
            }
        sharpenedImage.CopyTo(image);
    }
    // 3.0.3 shift image
    static void Shift(Span<float> sample, int offsetX, int offsetY)
    {
        int width = 32, height = 32;
        Span<float> shiftedSample = stackalloc float[sample.Length];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                int newX = x + offsetX, newY = y + offsetY;
                if (newX >= 0 && newX < width && newY >= 0 && newY < height)
                    shiftedSample[newY * width + newX] = sample[y * width + x];
            }
        shiftedSample.CopyTo(sample);
    }
    // 3.1.0 autofilter for test + training 
    static void ConvAutoFilterInputImage(Span<float> conv)
    {
        for (int i = 0; i < 3; i++)
            SharpResidualFilter(conv.Slice(i * 32 * 32, 32 * 32), 1);
    }
    // 3.1.1 sharpen + residual
    static void SharpResidualFilter(Span<float> image, float factor, int width = 32)
    {
        // Define the Laplacian kernel for edge detection
        float[] laplacianKernel = {
                factor*-1, factor* -1,  factor*-1,
                factor*-1,   factor* 8,  factor*-1,
                factor* -1,  factor*-1, factor* -1
            };
        Span<float> sharpenedImage = stackalloc float[image.Length];
        // Apply the Laplacian kernel to each pixel in the image
        for (int y = 1; y < width - 1; y++)
        {
            for (int x = 1; x < width - 1; x++)
            {
                float sum = 0;
                int kernelIndex = 0;
                for (int ky = -1; ky <= 1; ky++)
                {
                    for (int kx = -1; kx <= 1; kx++)
                    {
                        int pixelX = x + kx;
                        int pixelY = y + ky;
                        sum += image[pixelY * width + pixelX] * laplacianKernel[kernelIndex++];
                    }
                }
                // Apply the result to the current pixel
                float sharpPixel = image[y * width + x] + sum;
                sharpenedImage[y * width + x] = sharpPixel;
            }
        }
        sharpenedImage.CopyTo(image);
    }
    // 3.2 cnn ff
    static void ConvForward(int[] cnn, int[] dim, int[] cs, int[] filter, int[] kStep, int[] stride,
    Span<float> conv, Span<float> kernel, bool useAutoFilter = true)
    {
        if (useAutoFilter) ConvAutoFilterInputImage(conv);

        for (int i = 0; i < cnn.Length - 1; i++)
        {
            int left = cnn[i], right = cnn[i + 1], lDim = dim[i], rDim = dim[i + 1],
            lStep = cs[i + 0], rStep = cs[i + 1], kd = filter[i], ks = kStep[i], st = stride[i],
            lMap = lDim * lDim, rMap = rDim * rDim, kMap = kd * kd;

            if (kd == 3) // unrolled 3x3 kernel
            {
                for (int l = 0, leftStep = lStep; l < left; l++, leftStep += lMap) // input map
                {
                    for (int r = 0, rightStep = rStep; r < right; r++, rightStep += rMap) // out map
                    {
                        int kernelStep = ks + (l * right + r) * kMap;
                        var outMap = conv.Slice(rightStep, rMap);
                        for (int row = 0; row < kd; row++) // kernel filter dim y
                        {
                            var kernelcol = kernel.Slice(kernelStep + kd * row, kd);
                            float k1 = kernelcol[0], k2 = kernelcol[1], k3 = kernelcol[2];

                            for (int y = 0; y < rDim; y++) // conv dim y
                            {
                                var inputcol = conv.Slice(leftStep + (y * st + row) * lDim, lDim);
                                for (int x = 0; x < rDim; x++) // conv dim x
                                {
                                    int colStep = x * st;
                                    float sum1 = k1 * inputcol[colStep];
                                    float sum2 = k2 * inputcol[colStep + 1];
                                    float sum3 = k3 * inputcol[colStep + 2];
                                    outMap[x + y * rDim] += sum3 + sum2 + sum1;
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                // better cache locality?
                for (int l = 0, leftStep = lStep; l < left; l++, leftStep += lMap) // input map
                {
                    // if (dropoutMaps[layerMaps + l * right]) continue;
                    var inpMap = conv.Slice(leftStep, lDim * lDim);
                    for (int r = 0, rightStep = rStep; r < right; r++, rightStep += rMap) // out map
                    {
                        var outMap = conv.Slice(rightStep, rDim * rDim);
                        for (int row = 0; row < kd; row++) // kernel filter dim y
                        {
                            var kernelcol = kernel.Slice(ks + (l * right + r) * kMap + kd * row, kd);
                            for (int y = 0; y < rDim; y++) // conv dim y
                            {
                                var inputcol = inpMap.Slice((y * st + row) * lDim, lDim);
                                for (int x = 0; x < rDim; x++) // conv dim x
                                {
                                    float sum = 0; // kernel filter dim x
                                    for (int col = 0, colStep = x * st; col < kernelcol.Length; col++)
                                        sum = kernelcol[col] * inputcol[colStep + col] + sum;
                                    outMap[x + y * rDim] += sum;
                                }
                            }
                        }
                    }
                }
            }

            var activateMaps = conv.Slice(rStep, rDim * rDim * right);// relu activation
            ReLUSIMD(activateMaps);

        }
    }
    // 3.2.1 relu activation
    static void ReLUSIMD(Span<float> map)
    {
        var vec = MemoryMarshal.Cast<float, Vector512<float>>(map);
        for (int v = 0; v < vec.Length; v++) // SIMD
            vec[v] = Vector512.Max(vec[v], Vector512<float>.Zero);
        for (int k = vec.Length * Vector512<float>.Count; k < map.Length; k++)
            map[k] = MathF.Max(map[k], 0);
    }
    // 3.2.2 dropout between cnn output and nn input
    static void Dropout(int seed, Span<float> conv, int start, int part, float drop)
    {
        Random rng = new(seed);
        var outMap = conv.Slice(start, part);
        for (int k = 0; k < outMap.Length; k++) // conv map
        {
            float sum = outMap[k];
            if (sum > 0)
                outMap[k] = rng.NextSingle() > drop ? sum : 0;
        }
    }

    // 3.3 nn ff
    static void FeedForward(Span<int> net, Span<float> weights, Span<float> neurons)
    {
        for (int i = 0, k = net[0], w = 0; i < net.Length - 1; i++) // layers
        {
            // slice input + output layer
            var outLocal = neurons.Slice(k, net[i + 1]);
            var inpLocal = neurons.Slice(k - net[i], net[i]);
            for (int l = 0; l < inpLocal.Length; l++, w = outLocal.Length + w) // input neurons
            {
                // fast input neuron
                var inpNeuron = inpLocal[l];
                if (inpNeuron <= 0) continue; // ReLU input pre-activation
                                              // slice connected weights
                var wts = weights.Slice(w, outLocal.Length);
                // span to vector
                var wtsVec = MemoryMarshal.Cast<float, Vector512<float>>(wts);
                var outVec = MemoryMarshal.Cast<float, Vector512<float>>(outLocal);
                // SIMD
                for (int v = 0; v < outVec.Length; v++)
                    outVec[v] = wtsVec[v] * inpNeuron + outVec[v];
                // compute remaining output neurons
                for (int r = wtsVec.Length * Vector512<float>.Count; r < outLocal.Length; r++)
                    outLocal[r] = wts[r] * inpNeuron + outLocal[r];
            }
            k = outLocal.Length + k; // stack output id
        }
    }
    // 3.4 softmax
    static int SoftArgMax(Span<float> neurons, float temp = 1)
    {

        for (int n = 0; n < neurons.Length; n++)
            neurons[n] *= temp;

        int id = 0; // argmax
        float max = neurons[0];
        for (int i = 1; i < neurons.Length; i++)
            if (neurons[i] > max) { max = neurons[i]; id = i; }

        // softmax activation
        float scale = 0;
        for (int n = 0; n < neurons.Length; n++)
            scale += neurons[n] = MathF.Exp((neurons[n] - max));
        for (int n = 0; n < neurons.Length; n++)
            neurons[n] /= scale; // pseudo probabilities
        /*
        if (true) // LogSoftMaxWithMaxTrick
        {
            float scale = 0;
            for (int n = 0; n < neurons.Length; n++)
                if (neurons[n] < 0) neurons[n] = 0;

            for (int n = 0; n < neurons.Length; n++)
                scale += MathF.Exp(neurons[n] - max); // activation 

            scale = MathF.Log(scale); // log-softmax

            for (int n = 0; n < neurons.Length; n++)
                neurons[n] = MathF.Exp(neurons[n] - max - scale);
        }
        else
        {
            // softmax activation
            float scale = 0;
            for (int n = 0; n < neurons.Length; n++)
                scale += neurons[n] = MathF.Exp((neurons[n] - max));
            for (int n = 0; n < neurons.Length; n++)
                neurons[n] /= scale; // pseudo probabilities
        }
        */
        return id; // return nn prediction
    }
    // 3.5 output error gradient (target - output)
    static void ErrorGradient(Span<float> neurons, int target)
    {
        for (int i = 0; i < neurons.Length; i++)
            neurons[i] = target == i ? 1 - neurons[i] : -neurons[i];
    }
    // 3.6 nn bp
    static void Backprop(Span<float> neurons, Span<int> net, Span<float> weights, Span<float> deltas)
    {
        int j = neurons.Length - net[^1], k = neurons.Length, m = weights.Length;
        for (int i = net.Length - 2; i >= 0; i--)
        {
            int right = net[i + 1], left = net[i];
            k -= right; j -= left; m -= right * left;
            // slice input + output layer
            var inputNeurons = neurons.Slice(j, left);
            var outputGradients = neurons.Slice(k, right);
            for (int l = 0, w = m; l < left; l++, w += right)
            {
                var n = inputNeurons[l];
                if (n <= 0) { inputNeurons[l] = 0; continue; }
                // slice connected weights + deltas
                var wts = weights.Slice(w, right); // var inVec = Vector256.Create(n);
                var dts = deltas.Slice(w, right);
                // turn to vector
                var wtsVec = MemoryMarshal.Cast<float, Vector256<float>>(wts);
                var dtsVec = MemoryMarshal.Cast<float, Vector256<float>>(dts);
                var graVec = MemoryMarshal.Cast<float, Vector256<float>>(outputGradients);
                var sumVec = Vector256<float>.Zero;
                for (int v = 0; v < graVec.Length; v++) // SIMD, gradient sum and delta
                {
                    var outGraVec = graVec[v];
                    sumVec = wtsVec[v] * outGraVec + sumVec;
                    dtsVec[v] = n * outGraVec + dtsVec[v];
                }
                // turn vector sum to float
                var sum = Vector256.Sum(sumVec);
                // compute remaining elements
                for (int r = graVec.Length * Vector256<float>.Count; r < wts.Length; r++)
                {
                    var outGraSpan = outputGradients[r];
                    sum = wts[r] * outGraSpan + sum;
                    dts[r] = n * outGraSpan + dts[r];
                }
                inputNeurons[l] = sum; // reuse for gradients now
            }
        }
    }
    // 3.7 cnn bp
    static void ConvBackprop(int[] cnn, int[] dim, int[] cSteps, int[] filter, int[] kStep, int[] stride,
        Span<float> kernel, Span<float> kDelta, Span<float> cnnGradient, Span<float> conv)
    {
        for (int i = cnn.Length - 2; i >= 1; i--) // one loop bp: cnn gradient and kernel delta
        {
            int left = cnn[i], right = cnn[i + 1], lDim = dim[i], rDim = dim[i + 1],
            lStep = cSteps[i], rStep = cSteps[i + 1], kd = filter[i], ks = kStep[i], st =
            stride[i], lMap = lDim * lDim, rMap = rDim * rDim, kMap = kd * kd;
            for (int l = 0; l < left; l++, lStep += lMap) // input channel map
            {
                //   if (dropoutMaps[layerMaps + l * right]) continue;
                var inpMap = conv.Slice(lStep, lMap);
                var inpGraMap = cnnGradient.Slice(lStep, lMap);
                for (int r = 0, rs = rStep; r < right; r++, rs += rMap) // output channel map
                {
                    var outMap = conv.Slice(rs, rMap);
                    var graMap = cnnGradient.Slice(rs, rMap);
                    for (int row = 0; row < kd; row++) // filter dim y rows
                    {
                        int kernelID = ks + (l * right + r) * kMap + kd * row;
                        var kernelcol = kernel.Slice(kernelID, kd);
                        float k1 = kernelcol[0], k2 = kernelcol[1], k3 = kernelcol[2];
                        var kernelDeltacol = kDelta.Slice(kernelID, kd);
                        float kd1 = 0, kd2 = 0, kd3 = 0;

                        for (int y = 0; y < rDim; y++) // conv dim y
                        {
                            int irStep = (y * st + row) * lDim;
                            var inputcol = inpMap.Slice(irStep, lDim);
                            var inputGracol = inpGraMap.Slice(irStep, lDim);
                            int outStep = y * rDim;
                            for (int x = 0; x < rDim; x++) // conv dim x
                                if (outMap[x + outStep] > 0) // relu derivative
                                {
                                    int col = 0;
                                    float gra = graMap[x + outStep];

                                    int colStep = x * st;
                                    // Unrolled loop for 3 iterations
                                    kd1 += inputcol[colStep] * gra;
                                    inputGracol[colStep] += k1 * gra;
                                    col++;
                                    kd2 += inputcol[colStep + 1] * gra;
                                    inputGracol[colStep + 1] += k2 * gra;
                                    col++;
                                    kd3 += inputcol[colStep + 2] * gra;
                                    inputGracol[colStep + 2] += k3 * gra;
                                }
                        }
                        kernelDeltacol[0] += kd1;
                        kernelDeltacol[1] += kd2;
                        kernelDeltacol[2] += kd3;
                    }

                }
            }
        }
        {
            int i = 0;
            int left = cnn[i], right = cnn[i + 1], lDim = dim[i], rDim = dim[i + 1],
            lStep = cSteps[i], rStep = cSteps[i + 1], kd = filter[i], ks = kStep[i], st =
            stride[i], lMap = lDim * lDim, rMap = rDim * rDim, kMap = kd * kd;
            for (int l = 0; l < left; l++, lStep += lMap) // input channel map
            {
                //   if (dropoutMaps[layerMaps + l * right]) continue;
                var inpMap = conv.Slice(lStep, lMap);
                for (int r = 0, rs = rStep; r < right; r++, rs += rMap) // output channel map
                {
                    var outMap = conv.Slice(rs, rMap);
                    var graMap = cnnGradient.Slice(rs, rMap);
                    for (int row = 0; row < kd; row++) // filter dim y rows
                    {
                        int kernelID = ks + (l * right + r) * kMap + kd * row;
                        var kernelDeltacol = kDelta.Slice(kernelID, kd);
                        float kd1 = 0, kd2 = 0, kd3 = 0;

                        for (int y = 0; y < rDim; y++) // conv dim y
                        {
                            int irStep = (y * st + row) * lDim;
                            var inputcol = inpMap.Slice(irStep, lDim);
                            int outStep = y * rDim;
                            for (int x = 0; x < rDim; x++) // conv dim x
                                if (outMap[x + outStep] > 0) // relu derivative
                                {
                                    float gra = graMap[x + outStep];

                                    int colStep = x * st;
                                    // Unrolled loop for 3 iterations
                                    kd1 += inputcol[colStep] * gra;
                                    kd2 += inputcol[colStep + 1] * gra;
                                    kd3 += inputcol[colStep + 2] * gra;
                                }

                        }
                        kernelDeltacol[0] += kd1;
                        kernelDeltacol[1] += kd2;
                        kernelDeltacol[2] += kd3;
                    }
                }
            }
        }
    }
    // 3.8 sgd
    static void UpdateL2Norm(Span<float> weights, Span<float> delta, float lr, float mom, float lambda)
    {
        var lrNorm = lr * lambda;
        var weightVec = MemoryMarshal.Cast<float, Vector512<float>>(weights);
        var deltaVec = MemoryMarshal.Cast<float, Vector512<float>>(delta);
        // SIMD
        for (int v = 0; v < weightVec.Length; v++)
        {
            var wt = weightVec[v];
            weightVec[v] = deltaVec[v] * lr + wt - lrNorm * wt;
            deltaVec[v] *= mom;
        }

        for (int w = weightVec.Length * Vector512<float>.Count; w < weights.Length; w++)
        {
            var wt = weights[w];
            weights[w] = delta[w] * lr + wt - lrNorm * wt;
            delta[w] *= mom;
        }
    }

    // 4.0 train sample
    static int TrainSample(int id, int target, int seedD, int[] done, float[] data,
    int[] cnn, int[] dim, int[] filter, int[] stride, float[] kernel, float[] kernelDelta,
    int[] cSteps, int[] kStep, int[] net, float[] weight, float[] delta,
    int cnnLen, int nnLen, float drop, int epoch, bool flip, int shift, bool useAutoFilter, float randomFilter)
    {
        if (done[id] >= 2) return -1; // drop easy examples

        Random rng = new Random(seedD);

        // feed conv input layer with sample
        Span<float> conv = new float[cnnLen];
        int inputMap = dim[0] * dim[0];
        var sample = data.AsSpan().Slice(id * inputMap * cnn[0], inputMap * cnn[0]); // sample.CopyTo(conv);
        sample.CopyTo(conv.Slice(0, sample.Length));

        if (flip)
            if ((epoch + 1) % 2 == 0)
                // yes, no, maybe, maybe 
                // if ((epoch + 0) % 4 == 0 || (epoch + 2) % 4 == 0 && rng.NextSingle() < 0.5f  || (epoch + 3) % 4 == 0 && rng.NextSingle() > 0.5f)
                for (int i = 0; i < 3; i++)
                    Flip(conv.Slice(i * inputMap, inputMap));

        // RandomFilter     
        if (rng.NextSingle() < randomFilter)
        {
            // random kernel 
            Span<float> randomKernel = stackalloc float[9];
            for (int i = 0; i < 9; i++)
                randomKernel[i] = (0.5f - rng.NextSingle() * 1) * 0.5f;

            // int seedRandomFilter = random.Next();
            for (int i = 0; i < 3; i++)
            {
                var refImage = conv.Slice(i * inputMap, inputMap);
                RandomFilter(refImage, randomKernel);
            }
        }

        // Shift image
        if (shift != 0)
        {
            int offsetX = rng.Next(-shift, shift), offsetY = rng.Next(-shift, shift);
            if (offsetX != 0 && offsetX != 0) for (int i = 0; i < 3; i++) // rgb
                    Shift(conv.Slice(i * inputMap, inputMap), offsetX, offsetY);
        }

        // ff cnn
        ConvForward(cnn, dim, cSteps, filter, kStep, stride, conv, kernel, useAutoFilter);

        // dropout+: cnn + nn 
        if (drop > 0) Dropout(seedD, conv, cSteps[^2], cSteps[^1] - cSteps[^2], drop);

        // ff nn
        Span<float> neuron = new float[nnLen];
        conv.Slice(cSteps[^2], net[0]).CopyTo(neuron);
        FeedForward(net, weight, neuron);

        // nn prediction
        var outs = neuron.Slice(nnLen - net[^1], net[^1]);
        int prediction = SoftArgMax(outs);

        // drop easy examples
        if (outs[target] >= 0.9999) { done[id] += 1; return -1; }
        if (outs[target] >= 0.9) return prediction; // probability check

        // neural net backprop
        ErrorGradient(outs, target);
        Backprop(neuron, net, weight, delta);

        // conv net backprop
        Span<float> cnnGradient = new float[cSteps[^2] + net[0]];
        neuron.Slice(0, net[0]).CopyTo(cnnGradient.Slice(cSteps[^2], net[0]));
        ConvBackprop(cnn, dim, cSteps, filter, kStep, stride, kernel, kernelDelta, cnnGradient, conv);

        return prediction;
    }

    // 5.0 train epoch
    static float CnnTraining(bool multiCore, int[] indices, float[] data, byte[] label, float[] dataTest, byte[] labelTest,
    int seed, int[] done, int[] cnn, int[] filter, int[] stride, float[] kernel,
    float[] kernelDelta, int[] dim, int[] cSteps, int[] kStep, int[] net, float[] weight,
    float[] delta, int epoch, int batch, float lr, float l2, float mom,
    float drop, bool shuffle, bool flip, int shift, bool useAutoFilter, float randomFilter)
    {

        DateTime elapsed = DateTime.Now;
        Random rng = new Random(seed);

        if (shuffle) rng.Shuffle(indices);

        // get network size
        int cnnLen = CnnNeuronsLen(dim[0], cnn, dim), nnLen = NeuronsLen(net);

        int correct = 0, all = 0;

        // batch training 
        for (int b = 0, B = (int)(indices.Length / batch); b < B; b++) // each batch for one epoch
        {
            int[] seeds = Enumerable.Repeat(0, batch).Select(_ => rng.Next()).ToArray();

            if (multiCore) Parallel.For(0, batch, x => // each sample in this batch
            {
                // get shuffled supervised sample id
                int id = indices[x + b * batch], target = label[id];
                int prediction = TrainSample(id, target, seeds[x], done, data,
                cnn, dim, filter, stride, kernel, kernelDelta, cSteps,
                kStep, net, weight, delta, cnnLen, nnLen, drop, epoch, flip, shift, useAutoFilter, randomFilter);
                if (prediction != -1)
                {
                    if (prediction == target) Interlocked.Increment(ref correct);
                    Interlocked.Increment(ref all); // statistics accuracy
                }
            });
            else for (int x = 0; x < batch; x++)
                {
                    int id = indices[x + b * batch], target = label[id];
                    int prediction = TrainSample(id, target, seeds[x], done, data,
                    cnn, dim, filter, stride, kernel, kernelDelta, cSteps,
                    kStep, net, weight, delta, cnnLen, nnLen, drop, epoch, flip, shift, useAutoFilter, randomFilter);
                    if (prediction != -1)
                    {
                        if (prediction == target) Interlocked.Increment(ref correct);
                        Interlocked.Increment(ref all); // statistics accuracy
                    }
                }

            UpdateL2Norm(kernel, kernelDelta, lr, 0, l2); // no cnn mom works better?
            UpdateL2Norm(weight, delta, lr, mom, l2);

        }

        // cnn stuff
        int inputDim2d = dim[0] * dim[0];
        // correction value for each neural network weight
        float trainingAccuracy = (correct * 100.0f / all);
        correct = 0; all = 0;

        // test
        Parallel.For(0, labelTest.Length, id =>
        {
            int target = labelTest[id];
            Span<float> conv = new float[cnnLen];
            dataTest.AsSpan().Slice(id * inputDim2d * cnn[0], inputDim2d * cnn[0]).CopyTo(conv);
            // convolution feed forward
            ConvForward(cnn, dim, cSteps, filter, kStep, stride, conv, kernel);
            // copy cnn output to nn input
            Span<float> neuron = new float[nnLen];
            conv.Slice(cSteps[^2], net[0]).CopyTo(neuron);
            // neural net feed forward
            FeedForward(net, weight, neuron);

            var outs = neuron.Slice(nnLen - net[^1], net[^1]);

            int prediction = 0; // argmax
            float max = outs[0];
            for (int i = 1; i < outs.Length; i++)
                if (outs[i] > max) { max = outs[i]; prediction = i; }

            // statistics accuracy
            if (prediction == target)
                Interlocked.Increment(ref correct);
            Interlocked.Increment(ref all);
        });

        if ((epoch + 1) % 10 == 0)
            Console.WriteLine($"epoch = {(epoch + 1),3} | acc = {trainingAccuracy,6:F2}%"
            + $" | time = {(DateTime.Now - elapsed).TotalSeconds,5:F2}s"
            + $" | test = {(correct * 100.0 / all):F2}%");
        return (correct * 100.0f / all);
    }

    // 6.0 test cnn on unseen data
    static void CnnTesting(float[] data, byte[] label, int[] cnn, int[] filter, int[] stride,
    float[] kernel, int[] dim, int[] cSteps, int[] kStep, int[] net, float[] weight, int len, bool useAutoFilter)
    {
        DateTime elapsed = DateTime.Now;
        // cnn stuff
        int inputDim2d = dim[0] * dim[0];
        int cnnLen = CnnNeuronsLen(dim[0], cnn, dim), nnLen = NeuronsLen(net);
        // correction value for each neural network weight
        int correct = 0, all = 0;
        Parallel.For(0, len, id =>
        {
            int target = label[id];
            Span<float> conv = new float[cnnLen];
            data.AsSpan().Slice(id * inputDim2d * cnn[0], inputDim2d * cnn[0]).CopyTo(conv);
            // convolution feed forward
            ConvForward(cnn, dim, cSteps, filter, kStep, stride, conv, kernel);
            // copy cnn output to nn input
            Span<float> neuron = new float[nnLen];
            conv.Slice(cSteps[^2], net[0]).CopyTo(neuron);
            // neural net feed forward
            FeedForward(net, weight, neuron);

            var outs = neuron.Slice(nnLen - net[^1], net[^1]);

            int prediction = 0; // argmax
            float max = outs[0];
            for (int i = 1; i < outs.Length; i++)
                if (outs[i] > max) { max = outs[i]; prediction = i; }

            // statistics accuracy
            if (prediction == target)
                Interlocked.Increment(ref correct);
            Interlocked.Increment(ref all);
        });
        Console.WriteLine($"Test accuracy = {(correct * 100.0 / all):F2}%" +
        //    $" predicted = {all}, " + 
        $" after {(DateTime.Now - elapsed).TotalSeconds:F2}s");
    }

    public struct AutoCifar10
    {
        public float[] cifarTrainNorm;
        public byte[] cifarTrainbLabel;
        public float[] cifarTestNorm;
        public byte[] cifarTestLabel;

        public AutoCifar10(string path)

        {
            string targzFilePath = path + @"cifar-10-binary.tar.gz";
            // Create directory if it doesn't exist
            if (!Directory.Exists(path)) Directory.CreateDirectory(path);

            // Check if the targzFilePath exists
            if (!File.Exists(targzFilePath))
            {
                Console.WriteLine("CIFAR-10 not found: cifar-10-binary.tar.gz");
                Console.WriteLine("Download CIFAR-10 : (zip + extract = 338 MB)");

                string downloadUrl = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";

                if (!Directory.Exists(path)) Directory.CreateDirectory(path);

                // Download the file
                WebClient client = new WebClient();
                client.DownloadFile(downloadUrl, targzFilePath);

                //  Console.WriteLine($"Extract CIFAR-10 dataset:{path2 + @"cifar-10-batches-bin"}");
                ExtractTarGz(targzFilePath, path);
            }

            var (cifarTrainNorm2, cifarTrainbLabel2, cifarTestNorm2, cifarTestLabel2) = LoadCIFARData(path + "cifar-10-batches-bin");

            this.cifarTrainNorm = cifarTrainNorm2;
            this.cifarTrainbLabel = cifarTrainbLabel2;
            this.cifarTestNorm = cifarTestNorm2;
            this.cifarTestLabel = cifarTestLabel2;

            // Use the CIFAR data as needed
            Console.WriteLine("CIFAR-10 dataset loaded\n");
        }

        static (float[], byte[], float[], byte[]) LoadCIFARData(string path)
        {
            float[] cifarTrainNorm = new float[10000 * 3072 * 5];
            byte[] cifarTrainbLabel = new byte[10000 * 5];
            float[] cifarTestNorm = new float[10000 * 3072];
            byte[] cifarTestLabel = new byte[10000];

            // Load training data from batches
            for (int batch = 0; batch < 5; batch++)
            {
                var samplesCifar10 = File.ReadAllBytes(Path.Combine(path, $"data_batch_{batch + 1}.bin")).Take(10000 * 3073).ToArray();
                for (int id = 0, c = 0; id < 10000; id++)
                {
                    cifarTrainbLabel[batch * 10000 + id] = samplesCifar10[c++];
                    for (int pos = 0; pos < 3073 - 1; pos++, c++)
                        cifarTrainNorm[batch * 10000 * 3072 + id * 3072 + pos] = samplesCifar10[c] / 255.0f;
                }
            }

            // Load test data 
            var samplesTestCifar10 = File.ReadAllBytes(Path.Combine(path, "test_batch.bin")).Take(10000 * 3073).ToArray();
            for (int id = 0, c = 0; id < 10000; id++)
            {
                cifarTestLabel[id] = samplesTestCifar10[c++];
                for (int pos = 0; pos < 3073 - 1; pos++, c++)
                    cifarTestNorm[id * 3072 + pos] = samplesTestCifar10[c] / 255.0f;
            }

            return (cifarTrainNorm, cifarTrainbLabel, cifarTestNorm, cifarTestLabel);
        }

        static void ExtractTarGz(string filename, string outputDir)
        {
            using var stream = File.OpenRead(filename);
            int read;
            const int chunk = 4096;
            var buffer = new byte[chunk];
            using var gzip = new GZipStream(stream, CompressionMode.Decompress);
            using var ms = new MemoryStream();
            while ((read = gzip.Read(buffer, 0, buffer.Length)) > 0)
                ms.Write(buffer, 0, read);

            gzip.Close();
            ms.Seek(0, SeekOrigin.Begin);

            ExtractTar(ms, outputDir);

            static void ExtractTar(Stream stream, string outputDir)
            {
                var buffer = new byte[100];
                var longFileName = string.Empty;
                while (true)
                {
                    stream.Read(buffer, 0, 100);
                    var name = string.IsNullOrEmpty(longFileName) ? Encoding.ASCII.GetString(buffer).Trim('\0') : longFileName; //Use longFileName if we have one read
                    if (string.IsNullOrWhiteSpace(name))
                        break;
                    stream.Seek(24, SeekOrigin.Current);
                    stream.Read(buffer, 0, 12);
                    var size = Convert.ToInt64(Encoding.UTF8.GetString(buffer, 0, 12).Trim('\0').Trim(), 8);
                    stream.Seek(20, SeekOrigin.Current); //Move head to typeTag byte
                    var typeTag = stream.ReadByte();
                    stream.Seek(355L, SeekOrigin.Current); //Move head to beginning of data (byte 512)

                    if (typeTag == 'L')
                    {
                        //If Type Tag is 'L' we have a filename that is longer than the 100 bytes reserved for it in the header.
                        //We read it here and save it temporarily as it will be the file name of the next block where the actual data is
                        var buf = new byte[size];
                        stream.Read(buf, 0, buf.Length);
                        longFileName = Encoding.ASCII.GetString(buf).Trim('\0');
                    }
                    else
                    {
                        //We have a normal file or directory
                        longFileName = string.Empty; //Reset longFileName if current entry is not indicating one
                        var output = Path.Combine(outputDir, name);
                        if (!Directory.Exists(Path.GetDirectoryName(output)))
                            Directory.CreateDirectory(Path.GetDirectoryName(output));

                        if (!name.EndsWith("/")) //Directories are zero size and don't need anything written
                        {
                            using var str = File.Open(output, FileMode.OpenOrCreate, FileAccess.Write);
                            var buf = new byte[size];
                            stream.Read(buf, 0, buf.Length);
                            str.Write(buf, 0, buf.Length);
                        }
                    }

                    //Move head to next 512 byte block 
                    var pos = stream.Position;
                    var offset = 512 - (pos % 512);
                    if (offset == 512)
                        offset = 0;

                    stream.Seek(offset, SeekOrigin.Current);
                }
            }
        }
    }
} //

