// https://github.com/grensen/ML_demos

#if DEBUG
Console.WriteLine("Debug mode is on, switch to Release mode");
#endif

bool useUltimateNet = true;

string networkType = useUltimateNet ? "Ultimate ReLU-2D" : "Neural ReLU-2D";

Console.WriteLine($"\nBegin {networkType} Network Demo\n");

string path = @"C:\ultimate_relu2d_net\";
string filePath = path + @"ultimate_relu2d.txt";

AutoData d = new(path); // get data

// define ReLU-2D networkType 
int[] net = { 784, 64, 64, 64, 64, 64, 10 };
var LEARNINGRATE = 0.0008f;
var MOMENTUM = 0.9f;
var EPOCHS = 50;
var BATCHSIZE = 100;
var FACTOR = 0.99f;
var SEED = 1337;
var PARALLEL = true;

var ultimateWeightsPositive = CreateUltimateJaggedArray<float>(net, useUltimateNet);
var ultimateWeightsNegative = CreateUltimateJaggedArray<float>(net, useUltimateNet);
PrintHyperparameters(ultimateWeightsPositive, ultimateWeightsNegative);

WeightInit(net, ultimateWeightsPositive, SEED);
WeightInit(net, ultimateWeightsNegative, SEED);

var trained = RunTraining(PARALLEL, d, net, ultimateWeightsPositive, ultimateWeightsNegative,
    60000, LEARNINGRATE, MOMENTUM, FACTOR, EPOCHS, BATCHSIZE);

 RunUltimateTest(PARALLEL, d, trained.net, trained.weightsP, trained.weightsN, 10000);

Console.WriteLine($"\nSave {networkType} network to: " + filePath);
SaveUltimateNetToFile(filePath, trained.net, trained.weightsP, trained.weightsN);

Console.WriteLine($"Load {networkType} network from: " + filePath);
var net2 = LoadUltimateNetFromFile(filePath);

RunUltimateTest(PARALLEL, d, net2.net, net2.weightsP, net2.weightsN, 10000);
NetworkInfo(net2.net, net2.weightsP, net2.weightsN);

Console.WriteLine("End demo");
Console.ReadLine();
//+------------------------------------------------------------------------+

static (int[] net, float[][] weightsP, float[][] weightsN) RunTraining(bool multiCore, AutoData d,
    int[] net, float[][] weightsP, float[][] weightsN,
    int len, float lr, float mom, float FACTOR, int EPOCHS, int BATCHSIZE)
{
    System.Diagnostics.Stopwatch stopwatch = System.Diagnostics.Stopwatch.StartNew();

    Console.WriteLine($"\nStart Mini-Batch Training{(multiCore ? " - Parallel" : "")}:");

    float[][] deltasP = CreateUltimateJaggedArray<float>(net);
    float[][] deltasN = CreateUltimateJaggedArray<float>(net);

    int sourceWeightSize = GetWeightsSize(weightsP);
    int networkSize = net.Sum();
    int inputSize = net[0];

    Random rng = new Random(1337);

    int[] indices = new int[d.labelsTraining.Length];
    for (int i = 0; i < indices.Length; i++) 
        indices[i] = i;
          
    // run training
    for (int epoch = 0, B = len / BATCHSIZE; epoch < EPOCHS; epoch++, lr *= FACTOR, mom *= FACTOR)
    {
        bool[] c = new bool[B * BATCHSIZE]; // for proper parallel correct counts
        float[] ce = new float[B * BATCHSIZE]; // cross entropy loss

        Shuffle(indices, rng.Next()); 

        for (int b = 0; b < B; b++)
        {
            if (multiCore)
                Parallel.For(b * BATCHSIZE, (b + 1) * BATCHSIZE, x =>
                {
                    (c[indices[x]], ce[indices[x]]) = Train(
                        d.samplesTrainingF.AsSpan().Slice(indices[x] * inputSize, inputSize), d.labelsTraining[indices[x]],
                        net, weightsP, weightsN, deltasP, deltasN, networkSize);
                });
            else
                for (int x = b * BATCHSIZE, X = (b + 1) * BATCHSIZE; x < X; x++)
                    (c[indices[x]], ce[indices[x]]) = Train(d.samplesTrainingF.AsSpan().Slice(indices[x] * inputSize, inputSize),
                        d.labelsTraining[indices[x]], net, weightsP, weightsN, deltasP, deltasN, networkSize);

            SGD(weightsP, weightsN, deltasP, deltasN, lr, mom);
        }

        if ((epoch + 1) % 5 == 0)
            PrintInfo($"Epoch = {1 + epoch,3} | Loss = {CrossEntropyLoss(ce),6:F4} |", c.Count(n => n), B * BATCHSIZE, stopwatch);
    }

    return (net, weightsP, weightsN);
}
static (bool, float) Train(Span<float> sample, int target,
    int[] net, float[][] weightsP, float[][] weightsN, float[][] deltasP, float[][] deltasN, int networkSize)
{
    // ff
    var neurons = new float[networkSize];
    sample.CopyTo(neurons);
    FeedForward(net, neurons, weightsP, weightsN);
    // reference to output layer neurons
    var outs = neurons.AsSpan().Slice(neurons.Length - net[^1], net[^1]);
    int prediction = Argmax(outs);

    // bp
    Softmax(outs, outs[prediction]);
    if (outs[target] < 0.99f) // prediction prob lower than 99%
        Backprop(net, neurons, weightsP, weightsN, deltasP, deltasN, target);

    return (prediction == target, outs[target]);
}
static void RunUltimateTest(bool multiCore, AutoData d,
    int[] net, float[][] weightsP, float[][] weightsN, int len)
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
                net, weightsP, weightsN, networkSize);
        });
    else // single core
        for (int x = 0; x < len; x++)
            (c[x], ce[x]) = Test(d.samplesTestF.AsSpan().Slice(x * inputs, inputs), d.labelsTest[x],
                net, weightsP, weightsN, networkSize);

    stopwatch.Stop();

    PrintInfo($"\nTest | Loss = {CrossEntropyLoss(ce),6:F4} |", c.Count(n => n), len, stopwatch, true);
}
static (bool, float) Test(Span<float> sample, int target,
    int[] net, float[][] weightsP, float[][] weightsN, int networkSize)
{
    // ff
    var neurons = new float[networkSize];
    sample.CopyTo(neurons);
    FeedForward(net, neurons, weightsP, weightsN);
    // reference to output layer neurons
    var outs = neurons.AsSpan().Slice(neurons.Length - net[^1], net[^1]);
    int prediction = Argmax(outs);

    // needed for cross entropy
    Softmax(outs, outs[prediction]);

    return (prediction == target, outs[target]);
}

// 1. ultimate memory allocation
static T[][] CreateUltimateJaggedArray<T>(int[] net, bool useUltimate = true, bool useLogisticRegression = true)
{
    int networkSize = net.Sum();
    T[][] array = new T[networkSize - net[^1]][];

    // ultimate output neurons field: size - inputSize
    int field = net.Sum() - net[0];

    for (int i = 0, j = 0; i < net.Length - 1; i++)
    {
        if (useUltimate && !useLogisticRegression && i == 0)
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
static void WeightInit(int[] net, float[][] weights, int seed)
{
    Random rnd = new Random(seed);
    // 1. weights based on Glorot/Xavier initialization 
    for (int j = 0, k = net[0], i = 0; i < net.Length - 1; i++)
    {
        float sd = MathF.Sqrt(6.0f / (net[i] + weights[j + 0].Length));
        for (int l = 0; l < net[i]; l++)
            for (int r = 0; r < weights[j + l].Length; r++)
                weights[j + l][r] = (float)rnd.NextDouble() * sd - sd * 0.5f;
        j += net[i]; k += net[i + 1];
    }
}
// 3. feed forward
static void FeedForward(int[] net, float[] neurons, float[][] weightsP, float[][] weightsN)
{
    int k = 0;
    for (int i = 0; i < net.Length - 1; i++)
    {
        int j = k; k += net[i];
        for (int jl = j; jl < k; jl++)
        {
            float n = neurons[jl];
            if (n == 0) continue; // Pre-ReLU
            else if (n > 0) // +
                for (int R = weightsP[jl].Length, r = 0; r < R; r++)
                    neurons[k + r] += weightsP[jl][r] * n;
            else // -
                for (int R = weightsN[jl].Length, r = 0; r < R; r++)
                    neurons[k + r] += weightsN[jl][r] * n;
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
    if (false) // LogSoftMaxWithMaxTrick
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
static void Backprop(int[] net, float[] neurons, float[][] weightsP, float[][] weightsN, float[][] deltasP, float[][] deltasN, int target)
{
    // reusing neurons for gradient computation
    int outLayerStart = neurons.Length - net[^1];
    for (int r = outLayerStart, p = 0; r < neurons.Length; r++, p++)
        neurons[r] = target == p ? 1 - neurons[r] : -neurons[r];

    for (int j = outLayerStart, i = net.Length - 2; i >= 0; i--)
    {
        int k = j; j -= net[i];
        for (int jl = j; jl < k; jl++)
        {
            float n = neurons[jl];
            float sumGradient = 0;
            if (n == 0) continue; // Pre-ReLU
            else if (n > 0)  // +
                for (int R = weightsP[jl].Length, r = 0; r < R; r++)
                {
                    float gradient = neurons[k + r];
                    sumGradient += weightsP[jl][r] * gradient;
                    deltasP[jl][r] += n * gradient;
                }
            else  // - 
                for (int R = weightsN[jl].Length, r = 0; r < R; r++)
                {
                    float gradient = neurons[k + r];
                    sumGradient += weightsN[jl][r] * gradient;
                    deltasN[jl][r] += n * gradient;
                }
            neurons[jl] = sumGradient; // reusing neurons for gradient computation
        }
    }
}
// 5. update weights
static void SGD(float[][] weightsP, float[][] weightsN, float[][] deltasP, float[][] deltasN, float lr, float mom)
{
    // +
    for (int i = 0; i < weightsP.Length; i++)
        for (int j = 0; j < weightsP[i].Length; j++)
        {
            weightsP[i][j] += deltasP[i][j] * lr; 
            deltasP[i][j] *= mom;
        }
    // -
    for (int i = 0; i < weightsN.Length; i++)
        for (int j = 0; j < weightsN[i].Length; j++)
        {
            weightsN[i][j] += deltasN[i][j] * lr;
            deltasN[i][j] *= mom;
        }
}
// 6. save network weights and its positions  
static void SaveUltimateNetToFile(string fileName, int[] net, float[][] weightsP, float[][] weightsN)
{
    using (StreamWriter writer = new StreamWriter(fileName))
    {
        // Write the network architecture (net array)
        writer.WriteLine(string.Join(",", net));

        // Write the weights array
        foreach (var weight in weightsP)
            writer.WriteLine(string.Join(",", weight));

        // Write the weights array
        foreach (var weight in weightsN)
            writer.WriteLine(string.Join(",", weight));
    }
}
// 7. load network weights and its positions 
static (int[] net, float[][] weightsP, float[][] weightsN) LoadUltimateNetFromFile(string fileName)
{
    string[] lines = File.ReadAllLines(fileName);

    // 1. read the network
    int[] net = Array.ConvertAll(lines[0].Split(','), int.Parse);

    int half = (lines.Length - 1) / 2;

    float[][] weightsP = new float[half][];
    float[][] weightsN = new float[half][];
    // 2. read the weights
    for (int i = 1; i < half + 1; i++)
    {
        string line = lines[i];
        weightsP[i - 1] = line == "" ?
            Array.Empty<float>() : Array.ConvertAll(line.Split(','), float.Parse);
    }
    for (int i = 1 + half; i < lines.Length; i++)
    {
        string line = lines[i];
        weightsN[i - half - 1] = line == "" ?
            Array.Empty<float>() : Array.ConvertAll(line.Split(','), float.Parse);
    }
    return (net, weightsP, weightsN);
}

static int GetWeightsSize<T>(T[][] array)
{
    int size = 0;
    for (int i = 0; i < array.Length; i++)
        size += array[i].Length;
    return size;
}
static void Shuffle(int[] indices, int seed)
{
    Random random = new Random(seed);

    for (int i = 0; i < indices.Length; ++i)
    {
        int j = random.Next(i + 1);
        (indices[i], indices[j]) = (indices[j], indices[i]);
    }
}
static void NetworkInfo(int[] net, float[][] weightsP, float[][] weightsN)
{
    Console.WriteLine($"\nReLU-2D Network = {string.Join("-", net)} = {net.Sum()} (Weights: {GetWeightsSize(weightsP) + GetWeightsSize(weightsN)})\n");
}
void PrintHyperparameters(float[][] weightsP, float[][] weightsN)
{
    Console.WriteLine($"{networkType} Network Configuration:");
    Console.WriteLine("NETWORK      = " + string.Join("-", net));
    Console.WriteLine("WEIGHTS_Pos  = " + GetWeightsSize(weightsP));
    Console.WriteLine("WEIGHTS_Neg  = " + GetWeightsSize(weightsN));
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