// https://github.com/grensen/ML_demos
#if DEBUG
Console.WriteLine("Debug mode is on, switch to Release mode");
#endif

bool useUltimateNet = !true;

string networkType = useUltimateNet ? "Ultimate" : "Neural";

Console.WriteLine($"\nBegin ReLU {networkType} Network scale invariance demo\n");

string path = @"C:\relu_net\";
string filePath = path + @"relu_net.txt";

AutoData d = new(path); // get data

// define ultimate network 
int[] net = { 784, 100,100, 10 };
var LEARNINGRATE = 0.001f;
var MOMENTUM = 0.9f;
var EPOCHS = 50;
var BATCHSIZE = 50;
var FACTOR = 0.99f;
var SEED = 1337;
var PARALLEL = true;

var ultimateWeights = CreateUltimateJaggedArray<float>(net, useUltimateNet);

PrintHyperparameters(ultimateWeights);

WeightInit(net, ultimateWeights, SEED, useUltimateNet);

var trained = RunTraining(PARALLEL, d, net, ultimateWeights,
    60000, LEARNINGRATE, MOMENTUM, FACTOR, EPOCHS, BATCHSIZE);

RunUltimateTest(PARALLEL, d, trained.net, trained.weights, 10000);
Console.WriteLine($"Last weight original: {trained.weights[^1][^1]}");

float fac = 0.1f;
Console.WriteLine($"\nMultiply all weights by factor {fac}");
for (int i = 0; i < trained.weights.Length; i++)
    for (int j = 0; j < trained.weights[i].Length; j++)
        trained.weights[i][j] *= fac;

RunUltimateTest(PARALLEL, d, trained.net, trained.weights, 10000);
Console.WriteLine($"Last weight: {trained.weights[^1][^1]}");

fac = 100.0f;
Console.WriteLine($"\nMultiply all weights by factor {fac}");
for (int i = 0; i < trained.weights.Length; i++)
    for (int j = 0; j < trained.weights[i].Length; j++)
        trained.weights[i][j] *= fac;

RunUltimateTest(PARALLEL, d, trained.net, trained.weights, 10000);
Console.WriteLine($"Last weight: {trained.weights[^1][^1]}");

Console.WriteLine("\nEnd demo");
Console.ReadLine();
//+------------------------------------------------------------------------+

static (int[] net, float[][] weights) RunTraining(bool multiCore, AutoData d,
    int[] net, float[][] weights,
    int len, float lr, float mom, float FACTOR, int EPOCHS, int BATCHSIZE)
{
    System.Diagnostics.Stopwatch stopwatch = System.Diagnostics.Stopwatch.StartNew();

    Console.WriteLine($"\nStart Mini-Batch Training{(multiCore ? " - Parallel" : "")}:");

    int sourceWeightSize = GetWeightsSize(weights);
    int networkSize = net.Sum();
    int inputSize = net[0];
    float[] biasDeltas = new float[net.Sum()];
    float[][] deltas = CreateUltimateJaggedArray<float>(net);

    Random rng = new Random(1337);

    // run training
    for (int epoch = 0, B = len / BATCHSIZE, t = 0; epoch < EPOCHS; epoch++, lr *= FACTOR, mom *= FACTOR)
    {
        bool[] c = new bool[B * BATCHSIZE]; // for proper parallel correct counts
        float[] ce = new float[(len / BATCHSIZE) * BATCHSIZE]; // cross entropy loss
        
        int[] indices = Shuffle(d.labelsTraining.Length, rng.Next(), true);

        for (int b = 0; b < B; b++)
        {
            if (multiCore)
                Parallel.For(b * BATCHSIZE, (b + 1) * BATCHSIZE, x =>
                {
                    (c[indices[x]], ce[indices[x]]) = Train(
                        d.samplesTrainingF.AsSpan().Slice(indices[x] * inputSize, inputSize), d.labelsTraining[indices[x]],
                        net, weights, deltas, biasDeltas, networkSize);
                });
            else
                for (int x = b * BATCHSIZE, X = (b + 1) * BATCHSIZE; x < X; x++)
                    (c[indices[x]], ce[indices[x]]) = Train(d.samplesTrainingF.AsSpan().Slice(indices[x] * inputSize, inputSize),
                        d.labelsTraining[indices[x]], net, weights, deltas, biasDeltas, networkSize);

            SGD(weights, deltas, lr, mom);
        }

        if ((epoch + 1) % 10 == 0)
            PrintInfo($"Epoch = {1 + epoch,3} | Loss = {CrossEntropyLoss(ce),6:F4} |", c.Count(n => n), B * BATCHSIZE, stopwatch);
    }

    return (net, weights);
}
static (bool, float) Train(Span<float> sample, int target,
    int[] net, float[][] weights, float[][] deltas, float[] biasDeltas, int networkSize)
{
    // ff
    var neurons = new float[networkSize];
    sample.CopyTo(neurons);
    FeedForward(net, neurons, weights);
    // reference to output layer neurons
    var outs = neurons.AsSpan().Slice(neurons.Length - net[^1], net[^1]);
    int prediction = Argmax(outs);

    // bp
    Softmax(outs, outs[prediction]);
    if (outs[target] < 0.99f) // prediction prob lower than 99%
        Backprop(net, neurons, weights, deltas, target);

    //(predY[j] - actualY[j]) * (predY[j] - actualY[j]);

    return (prediction == target, outs[target]);
}
static void RunUltimateTest(bool multiCore, AutoData d,
    int[] net, float[][] weights, int len)
{
    System.Diagnostics.Stopwatch stopwatch = System.Diagnostics.Stopwatch.StartNew();

    bool[] c = new bool[len]; // for proper parallel count
    float[] ce = new float[len]; // cross entropy loss

    int networkSize = net.Sum();
    int inputs = net[0];

    if (multiCore)
        Parallel.For(0, len, x =>
        {
            (c[x], ce[x]) = Test(d.samplesTestF.AsSpan().Slice(x * inputs, inputs), d.labelsTest[x],
                net, weights, networkSize);
        });
    else // single core
        for (int x = 0; x < len; x++)
            (c[x], ce[x]) = Test(d.samplesTestF.AsSpan().Slice(x * inputs, inputs), d.labelsTest[x],
                net, weights, networkSize);

    stopwatch.Stop();

    PrintInfo($"\nTest | Loss = {CrossEntropyLoss(ce),6:F4} |", c.Count(n => n), len, stopwatch, true);
}
static (bool, float) Test(Span<float> sample, int target,
    int[] net, float[][] weights, int networkSize)
{
    // ff
    var neurons = new float[networkSize];
    sample.CopyTo(neurons);
    FeedForward(net, neurons, weights);
    // reference to output layer neurons
    var outs = neurons.AsSpan().Slice(neurons.Length - net[^1], net[^1]);
    int prediction = Argmax(outs);

    // needed for cross entropy
    Softmax(outs, outs[prediction]);

    return (prediction == target, outs[target]);
}

// 1. ultimate or neural memory allocation
static T[][] CreateUltimateJaggedArray<T>(int[] net, bool useUltimate = true, bool useLogisticRegression = true)
{
    int networkSize = net.Sum();
    T[][] array = new T[networkSize - net[^1]][];

    // ultimate output neurons field: size - inputSize
    int field = net.Sum() - net[0];

    for (int i = 0, j = 0; i < net.Length - 1; i++)
    {
        if (useUltimate && useLogisticRegression && i == 0)
        {
            for (int l = 0; l < net[i]; l++)
                array[j + l] =
                    new T[useUltimate ? field - net[^1] : net[i + 1]];
        }
        else
        {
            for (int l = 0; l < net[i]; l++)
                array[j + l] =
                    new T[useUltimate ? field : net[i + 1]];
        }
        j += net[i];
        field -= net[i + 1]; // substract next inputSize
    }

    return array;
}
// 2. user weights init
static void WeightInit(int[] net, float[][] weights, int seed, bool useUltimate = true)
{

    Random rnd = new Random(seed);

    if (!useUltimate)
    {
        for (int j = 0, k = net[0], i = 0; i < net.Length - 1; i++)
        {
            float sd = MathF.Sqrt(6.0f / (net[i] + weights[j + 0].Length));
            for (int l = 0; l < net[i]; l++)
                for (int r = 0; r < weights[j + l].Length; r++)
                    weights[j + l][r] = (float)rnd.NextDouble() * sd - sd * 0.5f;
            j += net[i]; k += net[i + 1];
        }
        return;
    }

    string[] methodNames = { "Zero", "Glorot", "Fixed range", "Input Connectivity", "Output Connectivity" };

    Console.WriteLine("\nChoose weights initialization method:");

    for (int i = 0; i < methodNames.Length; i++)
        Console.WriteLine($"{i} - {methodNames[i]}");

    int choice = int.Parse(Console.ReadLine());

    if (choice >= 0 && choice < methodNames.Length)
    {
        Console.WriteLine($"\nWeights initialization based on the chosen method: {methodNames[choice]}");



        switch (choice)
        {
            case 0:
                // 0. zero init
                for (int j = 0, k = net[0], i = 0; i < net.Length - 1; i++)
                {
                    for (int l = 0; l < net[i]; l++)
                        for (int r = 0; r < weights[j + l].Length; r++)
                            weights[j + l][r] = 0;
                    j += net[i]; k += net[i + 1];
                }
                break;
            case 1:
                // 1. weights based on Glorot/Xavier initialization 
                for (int j = 0, k = net[0], i = 0; i < net.Length - 1; i++)
                {
                    float sd = MathF.Sqrt(6.0f / (net[i] + weights[j + 0].Length));
                    for (int l = 0; l < net[i]; l++)
                        for (int r = 0; r < weights[j + l].Length; r++)
                            weights[j + l][r] = (float)rnd.NextDouble() * sd - sd * 0.5f;
                    j += net[i]; k += net[i + 1];
                }
                break;
            case 2:
                // 2. weights based on fixed range for each weight on each layer
                for (int i = 0, j = 0; i < net.Length - 1; i++)
                {
                    float sd = 0.01f;
                    for (int l = 0; l < net[i]; l++)
                        for (int r = 0; r < weights[j + l].Length; r++)
                            weights[j + l][r] = (float)rnd.NextDouble() * sd - sd * 0.5f;
                    j += net[i];
                }
                break;
            case 3:
                if (useUltimate)
                    // 3. weights based on input connectivity
                    for (int i = 0, j = 0, field = net.Sum() - net[0]; i < net.Length - 1; i++)
                    {
                        float sd = MathF.Sqrt(1.0f / (net[i] + field));
                        for (int l = 0; l < net[i]; l++)
                            for (int r = 0; r < field; r++)
                                weights[j + l][r] = (float)rnd.NextDouble() * sd - sd * 0.5f;
                        j += net[i];
                        field -= net[i + 1];
                    }
                break;
            case 4:
                if (useUltimate)
                    // 4. weights based on output connectivity
                    for (int i = 0, j = 0; i < net.Length - 1; i++)
                    {
                        int left = net[i];
                        for (int l = 0; l < left; l++)
                            for (int ii = i, c = 0, f = left; ii < net.Length - 1; ii++)
                            {
                                f += net[ii + 1];
                                float sd = MathF.Sqrt(1.0f / f);
                                for (int r = 0; r < net[ii + 1]; r++)
                                    weights[j + l][c++] = (float)rnd.NextDouble() * sd - sd * 0.5f;
                            }
                        j += left;
                    }
                break;
        }
    }
}
// 3. feed forward
static void FeedForward(int[] net, float[] neurons, float[][] weights, bool useBias = !false)
{
    int j = 0, k = net[0];
    // unroll input layer for non zero features
    for (int jl = j; jl < k; jl++)
    {
        float n = neurons[jl];
        if (n != 0)
            for (int R = weights[jl].Length, r = 0; r < R; r++)
                neurons[k + r] += weights[jl][r] * n;
    }
    for (int i = 1; i < net.Length - 1; i++)
    {
        j = k; k += net[i];
        for (int jl = j; jl < k; jl++)
        {
            float n = neurons[jl];
            if (n <= 0) continue; // Pre-ReLU
            for (int R = weights[jl].Length, r = 0; r < R; r++)
                neurons[k + r] += weights[jl][r] * n;
        }
    }
}
// 3.1. prediction
static int Argmax(Span<float> neurons)
{
    int id = 0;
    for (int i = 1; i < neurons.Length; i++)
        if (neurons[i] > neurons[id])
            id = i;
    return id; // prediction
}
// 3.2. probabilities
static void Softmax(Span<float> outs, float max)
{
    float scale = 0;
    if (true) // LogSoftmaxWithMaxTrick
    {
        for (int n = 0; n < outs.Length; n++)
            if (outs[n] < 0) outs[n] = 0;

        for (int n = 0; n < outs.Length; n++)
            scale += MathF.Exp(outs[n] - max); // activation 

        scale = MathF.Log(scale); // log-softmax

        for (int n = 0; n < outs.Length; n++)
            outs[n] = MathF.Exp(outs[n] - max - scale);
    }
    else
    {
        for (int n = 0; n < outs.Length; n++)
            scale += outs[n] = MathF.Exp(outs[n] - max); // activation and sum up

        scale = 1 / scale; // turns division to multiplication

        for (int n = 0; n < outs.Length; n++)
            outs[n] *= scale; // probabilities
    }

}
// 3.3. cross entropy loss
static float CrossEntropyLoss(float[] ce)
{
    float loss = 0;
    for (int i = 0; i < ce.Length; i++)
        loss += -MathF.Log(ce[i]);
    loss /= ce.Length;
    return loss;
}
// 4. backpropagation
static void Backprop(int[] net, float[] neurons, float[][] weight, float[][] deltas, int target)
{    
    // reusing neurons for gradient computation
    int outLayerStart = neurons.Length - net[^1];
    for (int r = outLayerStart, p = 0; r < neurons.Length; r++, p++)
        neurons[r] = target == p ? 1 - neurons[r] : -neurons[r];

    for (int j = outLayerStart, i = net.Length - 2; i >= 1; i--)
    {
        int k = j; j -= net[i];
        for (int jl = j; jl < k; jl++)
        {
            float n = neurons[jl]; 
            float sumGradient = 0;
            if (n > 0)  // Pre-ReLU derivative 
                for (int R = weight[jl].Length, r = 0; r < R; r++)
                {
                    float gradient = neurons[k + r];
                    sumGradient += weight[jl][r] * gradient;
                    deltas[jl][r] += n * gradient;
                }
            neurons[jl] = sumGradient; // reusing neurons for gradient computation
        }
    }
    // unroll input layer for non zero features
    for (int jl = 0, k = net[0]; jl < k; jl++)
    {
        float n = neurons[jl];
        if (n != 0)
            for (int R = weight[jl].Length, r = 0; r < R; r++)
                deltas[jl][r] += n * neurons[k + r];
    }
}
// 5. update weights
static void SGD(float[][] weights, float[][] deltas, float lr, float mom)
{
    for (int i = 0; i < weights.Length; i++)
        for (int j = 0; j < weights[i].Length; j++)
        {
           weights[i][j] += deltas[i][j] * lr;
           deltas[i][j] *= mom;
        }
}

static int GetWeightsSize<T>(T[][] array)
{
    int size = 0;
    for (int i = 0; i < array.Length; i++)
        size += array[i].Length;
    return size;
}
static int[] Shuffle(int length, int seed, bool doShuffle = true)
{
    Random random = new Random(seed);
    int[] indices = new int[length];
    for (int i = 0; i < length; i++) indices[i] = i;
    if (!doShuffle) return indices;
    for (int i = length - 1; i > 0; i--)
    {
        int j = random.Next(i + 1);
        (indices[i], indices[j]) = (indices[j], indices[i]);
    }
    return indices;
}
void PrintHyperparameters(float[][] weights)
{
    Console.WriteLine($"{networkType} Network Configuration:");
    Console.WriteLine("NETWORK      = " + string.Join("-", net));
    Console.WriteLine("WEIGHTS      = " + GetWeightsSize(weights));
    Console.WriteLine("SEED         = " + SEED);
    Console.WriteLine("LEARNINGRATE = " + LEARNINGRATE);
    Console.WriteLine("MOMENTUM     = " + MOMENTUM);
    Console.WriteLine("BATCHSIZE    = " + BATCHSIZE);
    Console.WriteLine("EPOCHS       = " + EPOCHS);
    Console.WriteLine("FACTOR       = " + FACTOR);
    Console.WriteLine("PARALLEL     = " + PARALLEL);
}
static void PrintInfo(string str, int correct, int all, System.Diagnostics.Stopwatch sw, bool showFPS = false)
{
    Console.Write($"{str} Accuracy = {(correct * 100.0 / all).ToString("F2").PadLeft(6)}% | " +
        // $"Correct = {correct:N0}/{all:N0} | " + 
        $"Time = {(sw.Elapsed.TotalMilliseconds / 1000.0).ToString("F3")}s");

    if (showFPS)
        Console.WriteLine($" | FPS = {10000 / sw.Elapsed.TotalSeconds:N0}");
    else
        Console.WriteLine();
}
// DATA
struct AutoData
{
    public byte[] labelsTraining, labelsTest;
    public float[] samplesTrainingF, samplesTestF;

    static float[] NormalizeData(byte[] samples)
    {
        float[] samplesF = new float[samples.Length];
        for (int i = 0; i < samples.Length; i++)
            samplesF[i] = samples[i] / 255f;
        return samplesF;
    }

    public AutoData(string yourPath)
    {
        // Hardcoded URLs from my GitHub
        string trainDataUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/train-images.idx3-ubyte";
        string trainLabelUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/train-labels.idx1-ubyte";
        string testDataUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/t10k-images.idx3-ubyte";
        string testLabelUrl = "https://github.com/grensen/gif_test/raw/master/MNIST_Data/t10k-labels.idx1-ubyte";

        byte[] test, training;

        // Change variable names for readability
        string trainDataPath = "trainData", trainLabelPath = "trainLabel", testDataPath = "testData", testLabelPath = "testLabel";

        if (!File.Exists(Path.Combine(yourPath, trainDataPath))
            || !File.Exists(Path.Combine(yourPath, trainLabelPath))
            || !File.Exists(Path.Combine(yourPath, testDataPath))
            || !File.Exists(Path.Combine(yourPath, testLabelPath)))
        {
            Console.WriteLine("Status: MNIST Dataset Not found");
            if (!Directory.Exists(yourPath)) Directory.CreateDirectory(yourPath);

            // Padding bits: data = 16, labels = 8
            Console.WriteLine("Action: Downloading and Cleaning the Dataset from GitHub");
            training = new HttpClient().GetAsync(trainDataUrl).Result.Content.ReadAsByteArrayAsync().Result.Skip(16).Take(60000 * 784).ToArray();
            labelsTraining = new HttpClient().GetAsync(trainLabelUrl).Result.Content.ReadAsByteArrayAsync().Result.Skip(8).Take(60000).ToArray();
            test = new HttpClient().GetAsync(testDataUrl).Result.Content.ReadAsByteArrayAsync().Result.Skip(16).Take(10000 * 784).ToArray();
            labelsTest = new HttpClient().GetAsync(testLabelUrl).Result.Content.ReadAsByteArrayAsync().Result.Skip(8).Take(10000).ToArray();

            Console.WriteLine("Save Path: " + yourPath + "\n");
            File.WriteAllBytesAsync(Path.Combine(yourPath, trainDataPath), training);
            File.WriteAllBytesAsync(Path.Combine(yourPath, trainLabelPath), labelsTraining);
            File.WriteAllBytesAsync(Path.Combine(yourPath, testDataPath), test);
            File.WriteAllBytesAsync(Path.Combine(yourPath, testLabelPath), labelsTest);
        }
        else
        {
            // Data exists on the system, just load from yourPath
            Console.WriteLine("Dataset: MNIST (" + yourPath + ")" + "\n");
            training = File.ReadAllBytes(Path.Combine(yourPath, trainDataPath)).Take(60000 * 784).ToArray();
            labelsTraining = File.ReadAllBytes(Path.Combine(yourPath, trainLabelPath)).Take(60000).ToArray();
            test = File.ReadAllBytes(Path.Combine(yourPath, testDataPath)).Take(10000 * 784).ToArray();
            labelsTest = File.ReadAllBytes(Path.Combine(yourPath, testLabelPath)).Take(10000).ToArray();
        }

        samplesTrainingF = NormalizeData(training);
        samplesTestF = NormalizeData(test);
    }
}