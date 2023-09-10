
#if DEBUG
Console.WriteLine("Debug mode is on, switch to Release mode");
#endif

Console.WriteLine($"\nBegin Ultimate ReLU-2D Network One-Shot Compression Pruning Demo\n");

string path = @"C:\ultimate_relu2d_pruning\";
string filePath = path + @"ultimate_relu2d_pruning.txt";

AutoData d = new(path); // get data

// define neural network 
int[] net = { 784, 32, 32, 32, 32, 10 };
var LEARNINGRATE = 0.00067f;
var MOMENTUM = 0.9f;
var EPOCHS = 50;
var BATCHSIZE = 100;
var FACTOR = 0.99f;
var PRUNING = 0.01f;
var NODEPROB = 0.3f;
var SEED = 1337;
var PARALLEL = false;

// ultimate or neural init possible 
var pruningPos = UltimatePruningNetworkMemoryAllocation(net);
var pruningNeg = UltimatePruningNetworkMemoryAllocation(net);

Print2DHyperparameters(pruningPos.weights, pruningNeg.weights);

PruningWeightInit(net, pruningPos.positions, pruningPos.weights, SEED);
PruningWeightInit(net, pruningNeg.positions, pruningNeg.weights, SEED);

var trained = RunPruningTraining(PARALLEL, d, net, 
    pruningPos.positions, pruningPos.weights, 
    pruningNeg.positions, pruningNeg.weights, 
    PRUNING, NODEPROB,
    60000, LEARNINGRATE, MOMENTUM, FACTOR, EPOCHS, BATCHSIZE);

RunPruningTest(PARALLEL, d, trained.net, trained.positionsPos, trained.weightsPos, trained.positionsNeg, trained.weightsNeg, 10000);

Console.WriteLine($"\nSave network to: " + filePath);
SaveUltimateRelu2DPruningNetToFile(filePath, trained.net, trained.positionsPos, trained.weightsPos, trained.positionsNeg, trained.weightsNeg);

Console.WriteLine($"Load network from: " + filePath);
var net2 = LoadUltimateRelu2DPruningNetFromFile(filePath);
RunPruningTest(PARALLEL, d, net2.net, net2.positionsPos, net2.weightsPos, net2.positionsNeg, net2.weightsNeg, 10000);

Console.WriteLine("\nEnd demo");
Console.ReadLine();

//+------------------------------------------------------------------------+

static (int[] net, int[][] positionsPos, float[][] weightsPos, int[][] positionsNeg, float[][] weightsNeg) RunPruningTraining(bool multiCore, AutoData d,
    int[] net, int[][] pPos, float[][] wPos, int[][] pNeg, float[][] wNeg, float pr, float np,
    int len, float lr, float mom, float FACTOR, int EPOCHS, int BATCHSIZE)
{

    System.Diagnostics.Stopwatch stopwatch = System.Diagnostics.Stopwatch.StartNew();

    Console.WriteLine($"\nStart Mini-Batch Training{(multiCore ? " - Parallel" : "")}:");

    float[][] deltasP = CreateUltimateJaggedArray<float>(net);
    float[][] deltasN = CreateUltimateJaggedArray<float>(net);

    int sourceWeightSize = GetWeightsSize(wPos);
    int sourceNetworkSize = net.Sum();
    int networkSize = net.Sum();
    int inputSize = net[0];

    Random rng = new(1337);

    // run training
    for (int epoch = 0, B = len / BATCHSIZE; epoch < EPOCHS; epoch++, lr *= FACTOR, mom *= FACTOR)
    {
        bool[] c = new bool[B * BATCHSIZE]; // for proper parallel correct counts
        float[] ce = new float[B * BATCHSIZE]; // cross entropy loss
        int[] indices = Shuffle(d.labelsTraining.Length, rng.Next(), false);

        for (int b = 0; b < B; b++)
        {
            if (multiCore)
                Parallel.For(b * BATCHSIZE, (b + 1) * BATCHSIZE, x =>
                {
                    (c[indices[x]], ce[indices[x]]) = PruningTrain(d.samplesTrainingF.AsSpan().Slice(indices[x] * inputSize, inputSize),
                        d.labelsTraining[indices[x]], net, pPos, wPos, pNeg, wNeg, deltasP, deltasN, networkSize);
                });
            else
                for (int x = b * BATCHSIZE, X = (b + 1) * BATCHSIZE; x < X; x++)
                    (c[indices[x]], ce[indices[x]]) = PruningTrain(d.samplesTrainingF.AsSpan().Slice(indices[x] * inputSize, inputSize),
                        d.labelsTraining[indices[x]], net, pPos, wPos, pNeg, wNeg, deltasP, deltasN, networkSize);

            // 5. sgd mini batch
            SGD2D(wPos, wNeg, deltasP, deltasN, lr, mom);
        }

        // 6. pruning
        if (epoch == EPOCHS / 2) 
            networkSize = CompressionPruning2D(ref net, ref pPos, ref wPos, ref deltasP, ref pNeg, ref wNeg, ref deltasN, pr, np);

        if ((epoch + 1) % 5 == 0)
            PrintInfo($"Epoch = {1 + epoch,3} | Loss = {CrossEntropyLoss(ce),6:F4} |", c.Count(n => n), B * BATCHSIZE, stopwatch);
    }

    return (net, pPos, wPos, pNeg, wNeg);
}
static (bool, float) PruningTrain(Span<float> sample, int target,
int[] net, int[][] pPos, float[][] wPos, int[][] pNeg, float[][] wNeg, float[][] deltasP, float[][] deltasN, int networkSize)
{
    // ff
    var neurons = new float[networkSize];
    sample.CopyTo(neurons);
    PruningFeedForward2D(net, neurons, pPos, wPos, pNeg, wNeg);

    // reference to output layer neurons
    var outs = neurons.AsSpan().Slice(neurons.Length - net[^1], net[^1]);
    int prediction = Argmax(outs);

    // bp
    Softmax(outs, outs[prediction]);
    if (outs[target] < 0.99f) // prediction probability lower than 99%
        PruningBackprop2D(net, neurons, pPos, wPos, pNeg, wNeg, deltasP, deltasN, target);

    return (target == prediction, outs[target]);
}


static void RunPruningTest(bool multiCore, AutoData d,
    int[] net, int[][] pPos, float[][] wPos, int[][] pNeg, float[][] wNeg, int len)
{
    System.Diagnostics.Stopwatch stopwatch = System.Diagnostics.Stopwatch.StartNew();

    bool[] c = new bool[len]; // for proper parallel count
    float[] ce = new float[len]; // cross entropy loss

    int networkSize = net.Sum();
    int inputs = net[0];
    
    if (multiCore)
        Parallel.For(0, len, x =>
        {
            (c[x], ce[x]) = PruningTest(d.samplesTestF.AsSpan().Slice(x * inputs, inputs), d.labelsTest[x],
                net, pPos, wPos, pNeg, wNeg, networkSize);
        });
    else // single core
        for (int x = 0; x < len; x++)
            (c[x], ce[x]) = PruningTest(d.samplesTestF.AsSpan().Slice(x * inputs, inputs), d.labelsTest[x],
                 net, pPos, wPos, pNeg, wNeg, networkSize);
    
    stopwatch.Stop();

    PrintInfo($"\nTest | Loss = {CrossEntropyLoss(ce),6:F4} |", c.Count(n => n), len, stopwatch, true);
}
static (bool, float) PruningTest(Span<float> sample, int target,
int[] net, int[][] pPos, float[][] wPos, int[][] pNeg, float[][] wNeg, int networkSize)
{
    // ff
    var neurons = new float[networkSize];
    sample.CopyTo(neurons);
    PruningFeedForward2D(net, neurons, pPos, wPos, pNeg, wNeg);

    // reference to output layer neurons
    var outs = neurons.AsSpan().Slice(neurons.Length - net[^1], net[^1]);
    int prediction = Argmax(outs);
    Softmax(outs, outs[prediction]);

    return (target == prediction, outs[target]);
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
static void PruningFeedForward2D(int[] net, float[] neurons, int[][] pPos, float[][] wPos, int[][] pNeg, float[][] wNeg)
{
    for (int k = 0, i = 0; i < net.Length - 1; i++)
    {
        int j = k; k += net[i];
        for (int jl = j; jl < k; jl++)
        {
            float n = neurons[jl];

            if (n == 0) continue;
            else if (n > 0)
                for (int R = wPos[jl].Length, r = 0; r < R; r++)
                {
                    int id = pPos[jl][r];
                    neurons[id] = wPos[jl][r] * n + neurons[id];
                }
            else
                for (int R = wNeg[jl].Length, r = 0; r < R; r++)
                {
                    int id = pNeg[jl][r];
                    neurons[id] = wNeg[jl][r] * n + neurons[id];
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
// 3.2. probabilities
static void Softmax(Span<float> outs, float max)
{
    float scale = 0;
    if (true) // LogSoftMaxWithMaxTrick
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
// 4.1. backpropagation
static void PruningBackprop2D(int[] net, float[] neurons, 
    int[][] pPos, float[][] wPos, int[][] pNeg, float[][] wNeg, float[][] deltasP, float[][] deltasN, int target)
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
            if (n == 0) continue;
            else if (n > 0)
                for (int R = wPos[jl].Length, r = 0; r < R; r++)
                {
                    float gra = neurons[pPos[jl][r]];
                    sumGradient += wPos[jl][r] * gra;
                    deltasP[jl][r] += n * gra;
                }
            else
                for (int R = wNeg[jl].Length, r = 0; r < R; r++)
                {
                    float gra = neurons[pNeg[jl][r]];
                    sumGradient += wNeg[jl][r] * gra;
                    deltasN[jl][r] += n * gra;
                }
            neurons[jl] = sumGradient; // reusing neurons for gradient computation
        }
    }
}
// 5. update weights
static void SGD2D(float[][] weightsP, float[][] weightsN, float[][] deltasP, float[][] deltasN, float lr, float mom)
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
// 6. compression pruning 
static int CompressionPruning2D(ref int[] net, ref int[][] pPos, ref float[][] wPos, ref float[][] deltasP, 
    ref int[][] pNeg, ref float[][] wNeg, ref float[][] deltasN, float pr, float np)
{
    int sourceNetworkSize = net.Sum();
    int sourceWeightSizeP = GetWeightsSize(wPos);
    int sourceWeightSizeN = GetWeightsSize(wNeg);
    Console.WriteLine($"\n1. Unstructured Magnitude Pruning: Wt² < {pr:F3}");
    UnstructuredPruning(net, pPos, wPos, deltasP, pr);
    UnstructuredPruning(net, pNeg, wNeg, deltasN, pr);

    NetworkPruning2DInfo(net, wPos, wNeg);

    Console.WriteLine($"2. Structured Compression Pruning: prunedLen / sourceLen <= {np:F2}");
    int networkSize = StructuredUltimatePruningRelu2D(ref net, ref pPos, ref wPos, ref deltasP, ref pNeg, ref wNeg, ref deltasN, np);

    NetworkPruning2DInfo(net, wPos, wNeg);

    int inputOutput = net[0] + net[^1];
    float compNet = (float)(networkSize - inputOutput) / (sourceNetworkSize - inputOutput) * 100;
    float compWeightsP = (100 - (float)GetWeightsSize(wPos) / sourceWeightSizeP * 100);
    float compWeightsN = (100 - (float)GetWeightsSize(wNeg) / sourceWeightSizeN * 100);
    Console.WriteLine($"Net Compression: {100 - compNet:F2}%, WeightsPos: {compWeightsP:F2}%, WeightsNeg: {compWeightsN:F2}%\n");

    // return new network size
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
static int StructuredUltimatePruningRelu2D(ref int[] net, ref int[][] pPos, ref float[][] wPos, ref float[][] deltasPos, 
    ref int[][] pNeg, ref float[][] wNeg, ref float[][] deltasNeg, float id)
{
    // 1. structured pruning based on low magnitude weights count
    bool[] removeNode = new bool[pPos.Length];
    int field = net.Sum() - net[0];
    for (int i = 0, j = 0; i < net.Length - 1; i++)
    {
        // input to center node
        for (int l = 0; l < net[i]; l++) //  if (((wPos[j + l].Length / (float)field) + (wNeg[j + l].Length / (float)field)) * 0.5 <= id)
            if (wPos[j + l].Length / (float)field <= id)
                if (wNeg[j + l].Length / (float)field <= id)
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
                for (int jl = 0; jl < pPos.Length; jl++)
                    for (int rrr = 0; rrr < pPos[jl].Length; rrr++)
                        if (pPos[jl][rrr] == idNode)
                        {
                            int last = pPos[jl].Length - 1;
                            for (int rr = rrr; rr < last; rr++)
                            {
                                pPos[jl][rr] = pPos[jl][rr + 1];
                                wPos[jl][rr] = wPos[jl][rr + 1];
                                deltasPos[jl][rr] = deltasPos[jl][rr + 1];
                            }
                            Array.Resize(ref pPos[jl], last);
                            Array.Resize(ref wPos[jl], last);
                            Array.Resize(ref deltasPos[jl], last);
                        }
                // 2.2 check ultimate weight connections to target node
                for (int jl = 0; jl < pNeg.Length; jl++)
                    for (int rrr = 0; rrr < pNeg[jl].Length; rrr++)
                        if (pNeg[jl][rrr] == idNode)
                        {
                            int last = pNeg[jl].Length - 1;
                            for (int rr = rrr; rr < last; rr++)
                            {
                                pNeg[jl][rr] = pNeg[jl][rr + 1];
                                wNeg[jl][rr] = wNeg[jl][rr + 1];
                                deltasNeg[jl][rr] = deltasNeg[jl][rr + 1];
                            }
                            Array.Resize(ref pNeg[jl], last);
                            Array.Resize(ref wNeg[jl], last);
                            Array.Resize(ref deltasNeg[jl], last);
                        }

                // 2.3 remove row as outgoing weights: weights, positions, deltas                    
                for (int kr = idNode; kr < wPos.Length - 1; kr++)
                {
                    wPos[kr] = wPos[kr + 1];
                    pPos[kr] = pPos[kr + 1];
                    deltasPos[kr] = deltasPos[kr + 1];
                }
                Array.Resize(ref wPos, wPos.Length - 1);
                Array.Resize(ref pPos, pPos.Length - 1);
                Array.Resize(ref deltasPos, deltasPos.Length - 1);

                // 2.3 remove row as outgoing weights: weights, positions, deltas                    
                for (int kr = idNode; kr < pNeg.Length - 1; kr++)
                {
                    pNeg[kr] = pNeg[kr + 1];
                    wNeg[kr] = wNeg[kr + 1];
                    deltasNeg[kr] = deltasNeg[kr + 1];
                }
                Array.Resize(ref pNeg, pNeg.Length - 1);
                Array.Resize(ref wNeg, wNeg.Length - 1);
                Array.Resize(ref deltasNeg, deltasNeg.Length - 1);

                // 2.4 decrease all positions more or equal to target node
                for (int jl = 0; jl < pPos.Length; jl++)
                    for (int rr = 0; rr < pPos[jl].Length; rr++)
                        if (pPos[jl][rr] >= idNode) pPos[jl][rr]--;

                for (int jl = 0; jl < pNeg.Length; jl++)
                    for (int rr = 0; rr < pNeg[jl].Length; rr++)
                        if (pNeg[jl][rr] >= idNode) pNeg[jl][rr]--;

                // 2.5 subtract dropped hidden neuron from network layer
                net[i + 1]--;

                // 2.6 neuron pruned, arrays shifted, stay in position
                r--;
            }
        }
        j += net[i]; k += net[i + 1];
    }

    // 3. return new network size
    return net.Sum();
}
// 7. save network weights and its positions  
static void SaveUltimateRelu2DPruningNetToFile(string fileName, int[] net, int[][] pPos, float[][] wPos, int[][] pNeg, float[][] wNeg)
{
    using (StreamWriter writer = new StreamWriter(fileName))
    {
        // 1. Write network 
        writer.WriteLine(string.Join(",", net));

        // 2. Write positive positions
        foreach (var position in pPos)
            writer.WriteLine(string.Join(",", position));

        // 3. Write positive weights
        foreach (var weight in wPos)
            writer.WriteLine(string.Join(",", weight));

        // set negative start identifier
        writer.WriteLine("#");

        // 4. Write negative positions
        foreach (var position in pNeg)
            writer.WriteLine(string.Join(",", position));

        // 5. Write negative weights
        foreach (var weight in wNeg)
            writer.WriteLine(string.Join(",", weight));
    }
}
// 8. load network weights and its positions 
static (int[] net, int[][] positionsPos, float[][] weightsPos, int[][] positionsNeg, float[][] weightsNeg) LoadUltimateRelu2DPruningNetFromFile(string fileName)
{
    string[] lines = File.ReadAllLines(fileName);

    // 0. split positive and negative parts
    int lengthUntilDelimiter = 1;
    foreach (string line in lines)
    {
        if (line == "#")
            break;
        lengthUntilDelimiter++;
    }

    // 1. read the network
    int[] net = Array.ConvertAll(lines[0].Split(','), int.Parse);
    
    // set parts
    int positiveSizeBoth = lengthUntilDelimiter - 2;
    int positiveSize = positiveSizeBoth / 2;
    int negativeSizeBoth = lines.Length - positiveSizeBoth - 2;
    int negativeSize = negativeSizeBoth / 2;

    int[][] positionsPos = new int[positiveSize][];
    float[][] weightsPos = new float[positiveSize][];
    int[][] positionsNeg = new int[negativeSize][];
    float[][] weightsNeg = new float[negativeSize][];

    // 2. read positive positions
    for (int i = 1, ii = 0; i < 1 + positiveSize; i++)
    {
        string line = lines[i];
        positionsPos[ii++] = line == "" ?
            Array.Empty<int>() : Array.ConvertAll(line.Split(','), int.Parse);
    }

    // 3. read positive weights
    for (int i = 1 + positiveSize, ii = 0; i < 1 + positiveSize * 2; i++)
    {
        string line = lines[i];
        weightsPos[ii++] = line == "" ?
            Array.Empty<float>() : Array.ConvertAll(line.Split(','), float.Parse);
    }
    // 4. read negative positions
    for (int i = 2 + positiveSizeBoth, ii = 0; i < 2 + positiveSizeBoth + negativeSize; i++)
    {
        string line = lines[i];
        positionsNeg[ii++] = line == "" ?
            Array.Empty<int>() : Array.ConvertAll(line.Split(','), int.Parse);
    }

    // 5. read negative weights
    for (int i = 2 + positiveSizeBoth + negativeSize, ii = 0; i < lines.Length; i++)
    {
        string line = lines[i];
        weightsNeg[ii++] = line == "" ?
            Array.Empty<float>() : Array.ConvertAll(line.Split(','), float.Parse);
    }

    return (net, positionsPos, weightsPos, positionsNeg, weightsNeg);
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
void Print2DHyperparameters(float[][] weightsP, float[][] weightsN)
{
    Console.WriteLine("Ultimate Network Configuration:");
    Console.WriteLine("NETWORK      = " + string.Join("-", net));
    Console.WriteLine("WEIGHTSP     = " + GetWeightsSize(weightsP));
    Console.WriteLine("WEIGHTSN     = " + GetWeightsSize(weightsN));
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
static void NetworkPruning2DInfo(int[] net, float[][] weightsP, float[][] weightsN)
{
    Console.WriteLine($"Network = {string.Join("-", net)} = {net.Sum()} (WeightsPos: {GetWeightsSize(weightsP)} | WeightsNeg: {GetWeightsSize(weightsN)})\n");
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
