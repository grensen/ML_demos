
#if DEBUG
    Console.WriteLine("Debug mode is on, switch to Release mode");
#endif

Console.WriteLine($"\nBegin Ultimate Network One-Shot Compression Pruning Demo\n");

string path = @"C:\ultimate_pruning_net\";
string filePath = path + @"ultimate_pruning_net.txt";

AutoData d = new(path); // get data

// define neural network 
int[] net = { 784, 50, 50, 10 };
var LEARNINGRATE = 0.0005f;
var MOMENTUM = 0.67f;
var EPOCHS = 50;
var BATCHSIZE = 800;
var FACTOR = 0.99f;
var PRUNING = 0.01f;
var NODEPROB = 0.4f;
var SEED = 1337;
var PARALLEL = false;

// ultimate or neural init possible 
var pruning = UltimatePruningNetworkMemoryAllocation(net);

PrintHyperparameters(pruning.weights);

PruningWeightInit(net, pruning.positions, pruning.weights, SEED);

var trained = RunPruningTraining(PARALLEL, d,
    net, pruning.positions, pruning.weights, PRUNING, NODEPROB,
    60000, LEARNINGRATE, MOMENTUM, FACTOR, EPOCHS, BATCHSIZE);

RunPruningTest(PARALLEL, d, trained.net, trained.positions, trained.weights, 10000);
NetworkPruningInfo(trained.net, trained.positions, trained.weights);


Console.WriteLine($"Save pruned network to: " + filePath);
SavePruningNetToFile(filePath, trained.net, trained.positions, trained.weights);

Console.WriteLine($"Load pruned network from: " + filePath);
var net2 = LoadPruningNetFromFile(filePath);

RunPruningTest(PARALLEL, d, net2.net, net2.positions, net2.weights, 10000);
NetworkPruningInfo(net2.net, net2.positions, net2.weights);

Console.WriteLine("End demo");
Console.ReadLine();
//+------------------------------------------------------------------------+

static (int[] net, int[][] positions, float[][] weights) RunPruningTraining(bool multiCore, AutoData d,
    int[] net, int[][] positions, float[][] weights, float pr, float np,
    int len, float lr, float mom, float FACTOR, int EPOCHS, int BATCHSIZE)
{

    System.Diagnostics.Stopwatch stopwatch = System.Diagnostics.Stopwatch.StartNew();

    Console.WriteLine($"\nStart Mini-Batch Training{(multiCore ? " - Parallel" : "")}:");

    float[][] deltas = CreateUltimateJaggedArray<float>(net);

    int sourceWeightSize = GetWeightsSize(weights);

    int networkSize = net.Sum();
    int inputSize = net[0];

    Random rng = new(1337);

    // run training
    for (int epoch = 0, B = len / BATCHSIZE; epoch < EPOCHS; epoch++, lr *= FACTOR, mom *= FACTOR)
    {

        bool[] c = new bool[B * BATCHSIZE]; // for proper parallel correct counts

        int[] indices = Shuffle(d.labelsTraining.Length, rng.Next(), false);

        for (int b = 0; b < B; b++)
        {
            if (multiCore)
                Parallel.For(b * BATCHSIZE, (b + 1) * BATCHSIZE, x =>
                {
                    c[x] = PruningTrain(
                        d.samplesTrainingF.AsSpan().Slice(indices[x] * inputSize, inputSize), d.labelsTraining[indices[x]],
                        net, positions, weights, deltas, networkSize);
                });
            else
                for (int x = b * BATCHSIZE, X = (b + 1) * BATCHSIZE; x < X; x++)
                    c[x] = PruningTrain(d.samplesTrainingF.AsSpan().Slice(indices[x] * inputSize, inputSize),
                        d.labelsTraining[indices[x]], net, positions, weights, deltas, networkSize);

            // 5. sgd mini batch
            SGD(weights, deltas, lr, mom);
        }

        // 6. pruning
        if (epoch == EPOCHS / 2) // if ((epoch + 1) % pruningEpoch == 1 && epoch != EPOCHS - 1 && epoch != 0)
        {
            networkSize = CompressionPruning(ref net, ref positions, ref weights, ref deltas, pr, np);
        }

        if ((epoch + 1) % 5 == 0)
        {
            int wSize = GetWeightsSize(weights);
            PrintInfo($"Epoch = {1 + epoch,3} | Weights = {((double)wSize / sourceWeightSize) * 100,6:F2}% |", c.Count(n => n), B * BATCHSIZE, stopwatch);
        }
    }

    return (net, positions, weights);
}
static bool PruningTrain(Span<float> sample, int target,
int[] net, int[][] positions, float[][] weights, float[][] deltas, int networkSize)
{
    // ff
    var neurons = new float[networkSize];
    sample.CopyTo(neurons);

    PruningFeedForward(net, neurons, positions, weights);
    // reference to output layer neurons
    var outs = neurons.AsSpan().Slice(neurons.Length - net[^1], net[^1]);
    int prediction = Argmax(outs);

    // bp
    Softmax(outs);
    if (outs[target] < 0.99f) // prediction probability lower than 99%
        PruningBackprop(net, neurons, positions, weights, deltas, target);

    return target == prediction;
}
static bool PruningTest(Span<float> sample, int target,
int[] net, int[][] positions, float[][] weights, int networkSize)
{
    // ff
    var neurons = new float[networkSize];
    sample.CopyTo(neurons);

    PruningFeedForward(net, neurons, positions, weights);

    // reference to output layer neurons
    var outs = neurons.AsSpan().Slice(neurons.Length - net[^1], net[^1]);
    int prediction = Argmax(outs);
    
    return target == prediction;
}
static void RunPruningTest(bool multiCore, AutoData d,
    int[] net, int[][] positions, float[][] weights, int len)
{
    System.Diagnostics.Stopwatch stopwatch = System.Diagnostics.Stopwatch.StartNew();

    bool[] c = new bool[len]; // for proper parallel count

    int networkSize = net.Sum();
    int inputs = net[0];

    if (multiCore)
        Parallel.For(0, len, x =>
        {
            c[x] = PruningTest(d.samplesTestF.AsSpan().Slice(x * inputs, inputs), d.labelsTest[x],
                net, positions, weights, networkSize);
        });
    else // single core
        for (int x = 0; x < len; x++)
            c[x] = PruningTest(d.samplesTestF.AsSpan().Slice(x * inputs, inputs), d.labelsTest[x],
                net, positions, weights, networkSize);

    stopwatch.Stop();

    PrintInfo("\nTest", c.Count(n => n), len, stopwatch, true);
}

// 1. memory allocation
static (int[][] positions, float[][] weights) UltimatePruningNetworkMemoryAllocation(int[] net)
{
    return (CreateUltimateJaggedArray<int>(net), CreateUltimateJaggedArray<float>(net));
}
// 1.1. ultimate network
static T[][] CreateUltimateJaggedArray<T>(int[] net, bool ultimateNet = true)
{
    int networkSize = net.Sum();
    T[][] array = new T[networkSize - net[^1]][];

    // extend field to inputs times all outputs 
    int field = net.Sum() - net[0];
    for (int i = 0, j = 0; i < net.Length - 1; i++)
    {
        for (int l = 0; l < net[i]; l++)
            array[j + l] = new T[ultimateNet ? field : net[i + 1]];
        j += net[i];
        field -= net[i + 1];
    }
    return array;
}
// 2. weights init
static void PruningWeightInit(int[] net, int[][] positions, float[][] weights, int seed)
{
    Random rnd = new Random(seed);

    // create random weights 
    for (int i = 0, j = 0, left = net[0]; i < net.Length - 1; i++)
    {
        int right = net[i + 1];
        float sd = MathF.Sqrt(6.0f / (net[i] + positions[j].Length));
        for (int l = 0; l < left; l++)
            for (int r = 0; r < positions[j + l].Length; r++)
            {
                positions[j + l][r] = j + left + r;
                weights[j + l][r] = (float)rnd.NextDouble() * sd - sd * 0.5f;
            }
        j += left;
        left = right;
    }
}
// 3. feed forward
static void PruningFeedForward(int[] net, float[] neurons, int[][] positions, float[][] weights)
{
    for (int k = 0, i = 0; i < net.Length - 1; i++)
    {
        int j = k; k += net[i];
        for (int jl = j; jl < k; jl++)
        {
            float n = neurons[jl];
            if (n <= 0) continue;
            for (int R = weights[jl].Length, r = 0; r < R; r++)
            {
                int id = positions[jl][r];
                neurons[id] = weights[jl][r] * n + neurons[id];
            }
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
// 4. probabilities
static void Softmax(Span<float> neurons)
{
    float scale = 0;
    for (int n = 0; n < neurons.Length; n++)
        scale += neurons[n] = MathF.Exp(neurons[n]); // activation then sum up
    for (int n = 0; n < neurons.Length; n++)
        neurons[n] /= scale; // probabilities
}
// 4.1. backpropagation
static void PruningBackprop(int[] net, float[] neurons, int[][] positions, float[][] weight, float[][] deltas, int target)
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
            float n = neurons[jl], sumGradient = 0;
            if (n > 0)
                for (int R = weight[jl].Length, r = 0; r < R; r++)
                {
                    float gra = neurons[positions[jl][r]];
                    sumGradient += weight[jl][r] * gra;
                    deltas[jl][r] += n * gra;
                }
            neurons[jl] = sumGradient; // reusing neurons for gradient computation
        }
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
// 6. compression pruning 
static int CompressionPruning(ref int[] net, ref int[][] positions, ref float[][] weights, ref float[][] deltas, float pr, float np)
{
    int sourceNetworkSize = net.Sum();
    int sourceWeightSize = GetWeightsSize(weights);

    Console.WriteLine($"\n1. Unstructured Magnitude Pruning: Wt² < {pr:F3}");
    UnstructuredPruning(net, positions, weights, deltas, pr);
    NetworkPruningInfo(net, positions, weights);

    Console.WriteLine($"2. Structured Compression Pruning: prunedLen / sourceLen <= {np:F2}");
    int networkSize = StructuredUltimatePruning(ref net, ref positions, ref weights, ref deltas, np);
    NetworkPruningInfo(net, positions, weights);

    // int networkSize = sourceNetworkSize;
    int inputOutput = net[0] + net[^1];
    float compNet = (float)(networkSize - inputOutput) / (sourceNetworkSize - inputOutput) * 100;
    float compWeights = (100 - (float)GetWeightsSize(weights) / sourceWeightSize * 100);
    Console.WriteLine($"Network Compression: {100 - compNet:F2}%, Weights Compression: {compWeights:F2}%\n");

    // 3. return new network size
    return networkSize;
}
// 6.1 unstructured pruning
static void UnstructuredPruning(int[] net, int[][] positions, float[][] weights, float[][] deltas, float pr)
{
    // 1. unstructured pruning
    for (int jl = 0; jl < positions.Length; jl++) // each input to outputs connected neuron (input + hidden)
        for (int r = 0; r < positions[jl].Length; r++) // each connected weight on all outputs
            if (weights[jl][r] * weights[jl][r] < pr) // magnitude pruning
            {

                // 1.1 shifts pruned weight to the end
                for (int rr = r; rr < positions[jl].Length - 1; rr++)
                {
                    positions[jl][rr] = positions[jl][rr + 1];
                    weights[jl][rr] = weights[jl][rr + 1];
                    deltas[jl][rr] = deltas[jl][rr + 1];
                }

                // 1.2 remove the pruned weight from column arrays
                Array.Resize(ref positions[jl], positions[jl].Length - 1);
                Array.Resize(ref weights[jl], weights[jl].Length - 1);
                Array.Resize(ref deltas[jl], deltas[jl].Length - 1);

                // 1.3 keep position r to re-check the shifted weights on rows
                r--;
            }
}
// 6.2 structured pruning 
static int StructuredUltimatePruning(ref int[] net, ref int[][] positions, ref float[][] weights, ref float[][] deltas, float id)
{
    // 1. structured pruning based on low magnitude weights count
    bool[] removeNode = new bool[positions.Length];
    int field = net.Sum() - net[0];
    for (int i = 0, j = 0; i < net.Length - 1; i++)
    {
        // input to center node
        for (int l = 0; l < net[i]; l++)
            if (weights[j + l].Length / (float)field <= id)
                removeNode[j + l] = true;

        j += net[i]; field -= net[i + 1];
    }

    int sourceCount = net[0]; // starts after input size
    // 2. remove hidden neuron nodes (node and its ingoing and outgoing wts)
    for (int i = 0, j = 0, k = net[0]; i < net.Length - 2; i++) // from first to last hidden layer
    {
        for (int r = 0; r < net[i + 1]; r++) // each hidden neuron output side
        {
            // 2.1 current hidden id, skip first input layer
            int idNode = k + r;

            if (removeNode[sourceCount++])
            {
                // 2.2 check ultimate weight connections to target node
                for (int jl = 0; jl < positions.Length; jl++)
                    for (int rrr = 0; rrr < positions[jl].Length; rrr++)
                        if (positions[jl][rrr] == idNode)
                        {    
                                int last = positions[jl].Length - 1;
                                for (int rr = rrr; rr < last; rr++)
                                {
                                    positions[jl][rr] = positions[jl][rr + 1];
                                    weights[jl][rr] = weights[jl][rr + 1];
                                    deltas[jl][rr] = deltas[jl][rr + 1];
                                }
                                Array.Resize(ref positions[jl], last);
                                Array.Resize(ref weights[jl], last);
                                Array.Resize(ref deltas[jl], last);
                        }

                // 2.3 remove row as outgoing weights: weights, positions, deltas                    
                for (int kr = idNode; kr < weights.Length - 1; kr++)
                {
                    weights[kr] = weights[kr + 1];
                    positions[kr] = positions[kr + 1];
                    deltas[kr] = deltas[kr + 1];
                }
                Array.Resize(ref weights, weights.Length - 1);
                Array.Resize(ref positions, positions.Length - 1);
                Array.Resize(ref deltas, deltas.Length - 1);

                // 2.4 decrease all positions more or equal to target node
                for (int jl = 0; jl < positions.Length; jl++)
                    for (int rr = 0; rr < positions[jl].Length; rr++)
                        if (positions[jl][rr] >= idNode) positions[jl][rr]--;

                // 2.5 subtract dropped hidden neuron from network layer
                net[i + 1]--;
                
                // 2.6 neuron pruned, arrays shifted, stay in position
                r--;

                // ez!!!!
            }
        }
        j += net[i]; k += net[i + 1];
    }

    // 3. return new network size
    return net.Sum();
}
// 7. save network weights and its positions  
static void SavePruningNetToFile(string fileName, int[] net, int[][] positions, float[][] weights)
{
    using (StreamWriter writer = new StreamWriter(fileName))
    {
        // Write the network architecture (net array)
        writer.WriteLine(string.Join(",", net));

        // Write the positions array
        foreach (var position in positions)
            writer.WriteLine(string.Join(",", position));

        // Write the weights array
        foreach (var weight in weights)
            writer.WriteLine(string.Join(",", weight));
    }
}
// 8. load network weights and its positions 
static (int[] net, int[][] positions, float[][] weights) LoadPruningNetFromFile(string fileName)
{
    string[] lines = File.ReadAllLines(fileName);

    // 1. read the network
    int[] net = Array.ConvertAll(lines[0].Split(','), int.Parse);

    int numPositionsLines = (lines.Length - 1) / 2;

    int[][] positions = new int[numPositionsLines][];
    float[][] weights = new float[numPositionsLines][];

    // 2. read the positions
    for (int i = 1; i < 1 + numPositionsLines; i++)
    {
        string line = lines[i];
        positions[i - 1] = line == "" ?
            Array.Empty<int>() : Array.ConvertAll(line.Split(','), int.Parse);
    }

    // 3. read the weights
    for (int i = 1 + numPositionsLines; i < lines.Length; i++)
    {
        string line = lines[i];
        weights[i - 1 - numPositionsLines] = line == "" ?
            Array.Empty<float>() : Array.ConvertAll(line.Split(','), float.Parse);
    }

    return (net, positions, weights);
}

// helper
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
    Console.WriteLine("Ultimate Network Configuration:");
    Console.WriteLine("NETWORK      = " + string.Join("-", net));
    Console.WriteLine("WEIGHTS      = " + GetWeightsSize(weights));
    Console.WriteLine("POSITIONS    = " + GetWeightsSize(weights));
    Console.WriteLine("SEED         = " + SEED);
    Console.WriteLine("PRUNINGRATE  = " + PRUNING);
    Console.WriteLine("NODEPROB     = " + NODEPROB);
    Console.WriteLine("LEARNINGRATE = " + LEARNINGRATE);
    Console.WriteLine("MOMENTUM     = " + MOMENTUM);
    Console.WriteLine("BATCHSIZE    = " + BATCHSIZE);
    Console.WriteLine("EPOCHS       = " + EPOCHS);
    Console.WriteLine("FACTOR       = " + FACTOR);
    Console.WriteLine("PARALLEL     = " + PARALLEL);
}
static void NetworkPruningInfo(int[] net, int[][] positions, float[][] weights)
{
    Console.WriteLine($"Network = {string.Join("-", net)} = {net.Sum()} (Weights: {GetWeightsSize(weights)} | Positions: {GetWeightsSize(positions)})\n");
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