// https://visualstudiomagazine.com/articles/2015/02/01/using-multi-swarm-training.aspx
// https://jamesmccaffrey.wordpress.com/2015/02/16/training-neural-networks-using-multi-swarm-optimization/

using System;

Console.WriteLine("\nBegin neural network with multi-swarm training demo");

int numInput = 4; // number features
int numHidden = 5;
int numOutput = 3; // number of classes for Y
int numRows = 1000;
int seed = 0;

Console.WriteLine($"\nGenerating {numRows} artificial data items with {numInput} features");
double[][] allData = MakeAllData(numInput, numHidden, numOutput, numRows, seed);

Console.WriteLine("\nCreating train (80%) and test (20%) matrices");
double[][] trainData, testData;
MakeTrainTest(allData, 0.80, seed, out trainData, out testData);

Console.WriteLine("\nTraining data:");
ShowData(trainData, 4, 2, true);
Console.WriteLine("Test data:");
ShowData(testData, 3, 2, true);

Console.WriteLine($"Creating a {numInput}-{numHidden}-{numOutput} neural network classifier");
NeuralNetwork nn = new NeuralNetwork(numInput, numHidden, numOutput);

int numSwarms = 4;
int numParticles = 4;
int maxEpochs = 10000;
Console.WriteLine("Setting numSwarms = " + numSwarms);
Console.WriteLine("Setting numParticles = " + numParticles);
Console.WriteLine("Setting maxEpochs = " + maxEpochs);
Console.WriteLine("\nStarting training");

double[] bestWeights = nn.Train(trainData, maxEpochs, numSwarms, numParticles);
Console.WriteLine("Training complete\n");
Console.WriteLine("Best weights found:");
ShowVector(bestWeights, 4, 10, true); // 4 decimals, 10 values per line

double trainAcc = nn.Accuracy(trainData, bestWeights);
Console.WriteLine("\nAccuracy on training data = " + trainAcc.ToString("F4"));

double testAcc = nn.Accuracy(testData, bestWeights);
Console.WriteLine("Accuracy on test data = " + testAcc.ToString("F4"));

Console.WriteLine("\nEnd neural network with multi-swarm demo\n");
Console.ReadLine();

static double[][] MakeAllData(int numInput, int numHidden, int numOutput, int numRows, int seed)
{
    Random rnd = new Random(seed);
    int numWeights = (numInput * numHidden) + numHidden +
      (numHidden * numOutput) + numOutput;
    double[] weights = new double[numWeights]; // actually weights & biases
    for (int i = 0; i < numWeights; ++i)
        weights[i] = 20.0 * rnd.NextDouble() - 10.0; // [-10.0 to +10.0]

    double[][] result = new double[numRows][]; // allocate return-result matrix
    for (int i = 0; i < numRows; ++i)
        result[i] = new double[numInput + numOutput]; // 1-of-N Y in last column

    double[][] ihWeights = new double[numInput][];
    for (int i = 0; i < numInput; ++i)
        ihWeights[i] = new double[numHidden];

    double[] hBiases = new double[numHidden];

    double[][] hoWeights = new double[numHidden][];
    for (int i = 0; i < numHidden; ++i)
        hoWeights[i] = new double[numOutput];

    double[] oBiases = new double[numOutput];

    // copy random wts in
    int k = 0; // pts into weights[]
    for (int i = 0; i < numInput; ++i)
        for (int j = 0; j < numHidden; ++j)
            ihWeights[i][j] = weights[k++];

    for (int i = 0; i < numHidden; ++i)
        hBiases[i] = weights[k++];

    for (int i = 0; i < numHidden; ++i)
        for (int j = 0; j < numOutput; ++j)
            hoWeights[i][j] = weights[k++];

    for (int i = 0; i < numOutput; ++i)
        oBiases[i] = weights[k++];

    for (int r = 0; r < numRows; ++r) // for each row
    {
        // generate random inputs
        double[] inputs = new double[numInput];
        for (int i = 0; i < numInput; ++i)
            inputs[i] = 20.0 * rnd.NextDouble() - 10.0; // [-10.0 to +10.0]

        // compute the outputs
        double[] outputs = new double[numOutput];
        double[] hSums = new double[numHidden]; // hidden nodes sums scratch array
        double[] oSums = new double[numOutput]; // output nodes sums
        double[] hOutputs = new double[numHidden];

        for (int j = 0; j < numHidden; ++j)  // compute i-h sum of weights * inputs
            for (int i = 0; i < numInput; ++i)
                hSums[j] += inputs[i] * ihWeights[i][j]; // note +=

        for (int i = 0; i < numHidden; ++i)  // add biases to input-to-hidden sums
            hSums[i] += hBiases[i];

        for (int i = 0; i < numHidden; ++i)   // apply hidden activation
            hOutputs[i] = HyperTan(hSums[i]); // hard-coded

        for (int j = 0; j < numOutput; ++j)   // compute h-o sum of weights * hOutputs
            for (int i = 0; i < numHidden; ++i)
                oSums[j] += hOutputs[i] * hoWeights[i][j];

        for (int i = 0; i < numOutput; ++i)  // add biases to input-to-hidden sums
            oSums[i] += oBiases[i];

        double[] softOut = Softmax(oSums); // softmax activation does all outputs at once for efficiency
        Array.Copy(softOut, outputs, softOut.Length);

        // translate outputs to 1-of-N
        double[] oneOfN = new double[numOutput]; // all 0.0
        int maxIndex = 0;
        double maxValue = outputs[0];
        for (int i = 0; i < numOutput; ++i)
            if (outputs[i] > maxValue)
            {
                maxIndex = i;
                maxValue = outputs[i];
            }
        oneOfN[maxIndex] = 1.0;

        // place input and 1-of-N output values into curr row
        int c = 0; // column into result[][]
        for (int i = 0; i < numInput; ++i) // inputs
            result[r][c++] = inputs[i];
        for (int i = 0; i < numOutput; ++i) // outputs
            result[r][c++] = oneOfN[i];
    } // each row
    return result;
} // MakeAllData
static double HyperTan(double x) // for MakeAllData
{
    if (x < -20.0) return -1.0; // approximation is correct to 30 decimals
    else if (x > 20.0) return 1.0;
    else return Math.Tanh(x);
}
static double[] Softmax(double[] oSums) // for MakeAllData
{
    double max = oSums[0];
    for (int i = 0; i < oSums.Length; ++i)
        if (oSums[i] > max) max = oSums[i];

    // determine scaling factor -- sum of exp(each val - max)
    double scale = 0.0;
    for (int i = 0; i < oSums.Length; ++i)
        scale += Math.Exp(oSums[i] - max);

    double[] result = new double[oSums.Length];
    for (int i = 0; i < oSums.Length; ++i)
        result[i] = Math.Exp(oSums[i] - max) / scale;

    return result; // now scaled so that xi sum to 1.0
}
static void MakeTrainTest(double[][] allData, double trainPct, int seed, out double[][] trainData, out double[][] testData)
{
    Random rnd = new Random(seed);
    int totRows = allData.Length;
    int numTrainRows = (int)(totRows * trainPct); // usually 0.80
    int numTestRows = totRows - numTrainRows;
    trainData = new double[numTrainRows][];
    testData = new double[numTestRows][];

    double[][] copy = new double[allData.Length][]; // ref copy of all data
    for (int i = 0; i < copy.Length; ++i)
        copy[i] = allData[i];

    for (int i = 0; i < copy.Length; ++i) // scramble order
    {
        int r = rnd.Next(i, copy.Length); // use Fisher-Yates
        double[] tmp = copy[r];
        copy[r] = copy[i];
        copy[i] = tmp;
    }
    for (int i = 0; i < numTrainRows; ++i)
        trainData[i] = copy[i];

    for (int i = 0; i < numTestRows; ++i)
        testData[i] = copy[i + numTrainRows];
} // MakeTrainTest
static void ShowData(double[][] data, int numRows, int decimals, bool indices)
{
    for (int i = 0; i < numRows; ++i)
    {
        if (indices == true)
            Console.Write("[" + i.ToString().PadLeft(2) + "]  ");
        for (int j = 0; j < data[i].Length; ++j)
        {
            double v = data[i][j];
            if (v >= 0.0)
                Console.Write(" "); // '+'
            Console.Write(v.ToString("F" + decimals) + "    ");
        }
        Console.WriteLine("");
    }
    Console.WriteLine(". . .");
    int lastRow = data.Length - 1;
    if (indices == true)
        Console.Write("[" + lastRow.ToString().PadLeft(2) + "]  ");
    for (int j = 0; j < data[lastRow].Length; ++j)
    {
        double v = data[lastRow][j];
        if (v >= 0.0)
            Console.Write(" "); // '+'
        Console.Write(v.ToString("F" + decimals) + "    ");
    }
    Console.WriteLine("\n");
}
static void ShowVector(double[] vector, int decimals, int lineLen, bool newLine)
{
    for (int i = 0; i < vector.Length; ++i)
    {
        if (i > 0 && i % lineLen == 0) Console.WriteLine("");
        if (vector[i] >= 0) Console.Write(" ");
        Console.Write(vector[i].ToString("F" + decimals) + " ");
    }
    if (newLine == true)
        Console.WriteLine("");
}

public class NeuralNetwork
{
    private int numInput; // number input nodes
    private int numHidden;
    private int numOutput;

    private double[] inputs;
    private double[][] ihWeights; // input-hidden
    private double[] hBiases;
    private double[] hOutputs;

    private double[][] hoWeights; // hidden-output
    private double[] oBiases;
    private double[] outputs;

    private Random rnd;

    public NeuralNetwork(int numInput, int numHidden, int numOutput)
    {
        this.numInput = numInput;
        this.numHidden = numHidden;
        this.numOutput = numOutput;
        this.inputs = new double[numInput];
        this.ihWeights = MakeMatrix(numInput, numHidden);
        this.hBiases = new double[numHidden];
        this.hOutputs = new double[numHidden];
        this.hoWeights = MakeMatrix(numHidden, numOutput);
        this.oBiases = new double[numOutput];
        this.outputs = new double[numOutput];
        this.rnd = new Random(0);
    } // ctor
    private static double[][] MakeMatrix(int rows, int cols) // helper for ctor
    {
        double[][] result = new double[rows][];
        for (int r = 0; r < result.Length; ++r)
            result[r] = new double[cols];
        return result;
    }
    public double[] Train(double[][] trainData, int maxEpochs, int numSwarms, int numParticles)
    {
        int dim = (numInput * numHidden) + numHidden +
          (numHidden * numOutput) + numOutput;
        double minX = -9.9999;
        double maxX = 9.9999;

        MultiSwarm ms = new MultiSwarm(numSwarms, numParticles, dim); // array of arrays of Particles

        // supply particle errors - Error() not visible to particle ctor
        for (int i = 0; i < numSwarms; ++i)
        {
            for (int j = 0; j < numParticles; ++j)
            {
                Particle p = ms.swarms[i].particles[j];
                p.error = MeanSquaredError(trainData, p.position);
                p.bestError = p.error;
                Array.Copy(p.position, p.bestPosition, dim);
                if (p.error < ms.swarms[i].bestError) // swarm best?
                {
                    ms.swarms[i].bestError = p.error;
                    Array.Copy(p.position, ms.swarms[i].bestPosition, dim);
                }
                if (p.error < ms.bestError) // global best?
                {
                    ms.bestError = p.error;
                    Array.Copy(p.position, ms.bestPosition, dim);
                }
            } // j
        } // i

        int epoch = 0;
        double w = 0.729; // inertia
        double c1 = 1.49445; // particle-cogntive
        double c2 = 1.49445; // swarm-social
        double c3 = 0.3645; // multiswarm-global
                            //double pDeath = 1.0 / maxEpochs;      // prob of particle death
        double maxProbDeath = 0.10;
        double pImmigrate = 1.0 / maxEpochs;  // prob of particle immigration

        int[] sequence = new int[numParticles]; // process particles in random order
        for (int i = 0; i < sequence.Length; ++i)
            sequence[i] = i;

        while (epoch < maxEpochs)
        {
            ++epoch;

            if (epoch % (maxEpochs / 10) == 0 && epoch < maxEpochs)
            {
                Console.Write("Epoch = " + epoch);
                Console.WriteLine("   Best error = " + ms.bestError.ToString("F4"));
            }

            for (int i = 0; i < numSwarms; ++i) // each swarm
            {
                Shuffle(sequence); // move particles in random sequence
                for (int pj = 0; pj < numParticles; ++pj) // each particle
                {
                    int j = sequence[pj];
                    Particle p = ms.swarms[i].particles[j]; // curr particle ref

                    // 1. update velocity
                    for (int k = 0; k < dim; ++k)
                    {
                        double r1 = rnd.NextDouble();
                        double r2 = rnd.NextDouble();
                        double r3 = rnd.NextDouble();
                        p.velocity[k] = (w * p.velocity[k]) +
                          (c1 * r1 * (p.bestPosition[k] - p.position[k])) +
                          (c2 * r2 * (ms.swarms[i].bestPosition[k] - p.position[k])) +
                          (c3 * r3 * (ms.bestPosition[k] - p.position[k]));

                        if (p.velocity[k] < minX)
                            p.velocity[k] = minX;
                        else if (p.velocity[k] > maxX)
                            p.velocity[k] = maxX;
                    } // k

                    // 2. update position
                    for (int k = 0; k < dim; ++k)
                    {
                        p.position[k] += p.velocity[k];
                        // constrain all xi in new position
                        if (p.position[k] < minX)
                            p.position[k] = minX;
                        else if (p.position[k] > maxX)
                            p.position[k] = maxX;
                    }

                    // 3. update error at new position
                    p.error = MeanSquaredError(trainData, p.position); // expensive

                    // 4. update best particle, swarm, and best global error if necessary
                    if (p.error < p.bestError) // new best position for particle?
                    {
                        p.bestError = p.error;
                        Array.Copy(p.position, p.bestPosition, dim);
                    }
                    if (p.error < ms.swarms[i].bestError) // new best for swarm?
                    {
                        ms.swarms[i].bestError = p.error;
                        Array.Copy(p.position, ms.swarms[i].bestPosition, dim);
                    }
                    if (p.error < ms.bestError) // new global best?
                    {
                        ms.bestError = p.error;
                        Array.Copy(p.position, ms.bestPosition, dim);
                    }

                    // 5. die?
                    double p1 = rnd.NextDouble();
                    double pDeath = (maxProbDeath / maxEpochs) * epoch; // pDeath increases over time
                    if (p1 < pDeath)
                    {
                        Particle q = new Particle(dim); // a replacement
                        q.error = MeanSquaredError(trainData, q.position);
                        Array.Copy(q.position, q.bestPosition, dim);
                        q.bestError = q.error;

                        if (q.error < ms.swarms[i].bestError) // new best swarm error by pure luck?
                        {
                            ms.swarms[i].bestError = q.error;
                            Array.Copy(q.position, ms.swarms[i].bestPosition, dim);

                            if (q.error < ms.bestError) // best global error?
                            {
                                ms.bestError = q.error;
                                Array.Copy(q.position, ms.bestPosition, dim);
                            }
                        }

                        ms.swarms[i].particles[j] = q; // die!
                    } // die

                    // 6. immigrate?
                    double p2 = rnd.NextDouble();
                    if (p2 < pImmigrate)
                    {
                        int ii = rnd.Next(0, numSwarms); // rnd swarm
                        int jj = rnd.Next(0, numParticles); // rnd particle
                        Particle q = ms.swarms[ii].particles[jj]; // q points to other

                        ms.swarms[i].particles[j] = q;
                        ms.swarms[ii].particles[jj] = p; // the exchange
                                                         // p now points to other, q points to curr

                        if (q.error < ms.swarms[i].bestError) // curr has new position
                        {
                            ms.swarms[i].bestError = q.error;
                            Array.Copy(q.position, ms.swarms[i].bestPosition, dim);
                        }
                        if (p.error < ms.swarms[ii].bestError) // other has new position
                        {
                            ms.swarms[ii].bestError = p.error;
                            Array.Copy(p.position, ms.swarms[ii].bestPosition, dim);
                        }
                    } // immigrate

                } // j
            } // i

        } // while

        this.SetWeights(ms.bestPosition);
        return ms.bestPosition;
    } // Train
    private void Shuffle(int[] sequence) // instance so can use this.rnd
    {
        for (int i = 0; i < sequence.Length; ++i)
        {
            int r = rnd.Next(i, sequence.Length);
            int tmp = sequence[r];
            sequence[r] = sequence[i];
            sequence[i] = tmp;
        }
    }
    public void SetWeights(double[] weights)
    {
        // copy weights and biases in weights[] array to i-h weights, i-h biases, h-o weights, h-o biases
        int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
        if (weights.Length != numWeights)
            throw new Exception("Bad weights array in SetWeights");

        int k = 0; // points into weights param

        for (int i = 0; i < numInput; ++i)
            for (int j = 0; j < numHidden; ++j)
                ihWeights[i][j] = weights[k++];
        for (int i = 0; i < numHidden; ++i)
            hBiases[i] = weights[k++];
        for (int i = 0; i < numHidden; ++i)
            for (int j = 0; j < numOutput; ++j)
                hoWeights[i][j] = weights[k++];
        for (int i = 0; i < numOutput; ++i)
            oBiases[i] = weights[k++];
    }
    public double[] ComputeOutputs(double[] xValues)
    {
        double[] hSums = new double[numHidden]; // hidden nodes sums scratch array
        double[] oSums = new double[numOutput]; // output nodes sums

        for (int i = 0; i < xValues.Length; ++i) // copy x-values to inputs
            this.inputs[i] = xValues[i];

        for (int j = 0; j < numHidden; ++j)  // compute i-h sum of weights * inputs
            for (int i = 0; i < numInput; ++i)
                hSums[j] += this.inputs[i] * this.ihWeights[i][j]; // note +=

        for (int i = 0; i < numHidden; ++i)  // add biases to input-to-hidden sums
            hSums[i] += this.hBiases[i];

        for (int i = 0; i < numHidden; ++i)   // apply activation
            this.hOutputs[i] = HyperTan(hSums[i]); // hard-coded

        for (int j = 0; j < numOutput; ++j)   // compute h-o sum of weights * hOutputs
            for (int i = 0; i < numHidden; ++i)
                oSums[j] += hOutputs[i] * hoWeights[i][j];

        for (int i = 0; i < numOutput; ++i)  // add biases to input-to-hidden sums
            oSums[i] += oBiases[i];

        double[] softOut = Softmax(oSums); // softmax activation does all outputs at once for efficiency
        Array.Copy(softOut, outputs, softOut.Length);

        double[] retResult = new double[numOutput]; // could define a GetOutputs method instead
        Array.Copy(this.outputs, retResult, retResult.Length);
        return retResult;
    }
    private static double HyperTan(double x)
    {
        if (x < -20.0) return -1.0; // approximation is correct to 30 decimals
        else if (x > 20.0) return 1.0;
        else return Math.Tanh(x);
    }
    private static double[] Softmax(double[] oSums)
    {
        // does all output nodes at once so scale doesn't have to be re-computed each time
        // determine max output-sum
        double max = oSums[0];
        for (int i = 0; i < oSums.Length; ++i)
            if (oSums[i] > max) max = oSums[i];

        // determine scaling factor -- sum of exp(each val - max)
        double scale = 0.0;
        for (int i = 0; i < oSums.Length; ++i)
            scale += Math.Exp(oSums[i] - max);

        double[] result = new double[oSums.Length];
        for (int i = 0; i < oSums.Length; ++i)
            result[i] = Math.Exp(oSums[i] - max) / scale;

        return result; // now scaled so that xi sum to 1.0
    }
    public double MeanSquaredError(double[][] trainData, double[] weights)
    {
        this.SetWeights(weights); // copy the weights to evaluate in

        double[] xValues = new double[numInput]; // inputs
        double[] tValues = new double[numOutput]; // targets
        double sumSquaredError = 0.0;
        for (int i = 0; i < trainData.Length; ++i) // walk through each training data item
        {
            // following assumes data has all x-values first, followed by y-values!
            Array.Copy(trainData[i], xValues, numInput); // extract inputs
            Array.Copy(trainData[i], numInput, tValues, 0, numOutput); // extract targets
            double[] yValues = this.ComputeOutputs(xValues);
            for (int j = 0; j < yValues.Length; ++j)
                sumSquaredError += ((yValues[j] - tValues[j]) * (yValues[j] - tValues[j]));
        }
        return sumSquaredError / trainData.Length;
    }
    public double Accuracy(double[][] testData, double[] weights)
    {
        this.SetWeights(weights);
        // percentage correct using winner-takes all
        int numCorrect = 0;
        int numWrong = 0;
        double[] xValues = new double[numInput]; // inputs
        double[] tValues = new double[numOutput]; // targets
        double[] yValues; // computed Y

        for (int i = 0; i < testData.Length; ++i)
        {
            Array.Copy(testData[i], xValues, numInput); // parse data into x-values and t-values
            Array.Copy(testData[i], numInput, tValues, 0, numOutput);
            yValues = this.ComputeOutputs(xValues);
            int maxIndex = MaxIndex(yValues); // which cell in yValues has largest value?

            if (tValues[maxIndex] == 1.0) // ugly. consider AreEqual(double x, double y, double epsilon)
                ++numCorrect;
            else
                ++numWrong;
        }
        return (numCorrect * 1.0) / (numCorrect + numWrong); // ugly 2 - check for divide by zero
    }
    private static int MaxIndex(double[] vector) // helper for Accuracy()
    {
        // index of largest value
        int bigIndex = 0;
        double biggestVal = vector[0];
        for (int i = 0; i < vector.Length; ++i)
            if (vector[i] > biggestVal)
            {
                biggestVal = vector[i];
                bigIndex = i;
            }
        return bigIndex;
    }

    // ===== nested classes for MSO training =====
    private class Particle
    {
        static Random rnd = new Random(0); // shared by all instances. could parameterize seed
        public double[] position;
        public double[] velocity;
        public double error;
        public double[] bestPosition;
        public double bestError;
        private double minX = -9.9999;
        private double maxX = 9.9999;

        public Particle(int dim)
        {
            position = new double[dim];
            velocity = new double[dim];
            bestPosition = new double[dim];
            for (int k = 0; k < dim; ++k)
            {
                position[k] = (maxX - minX) * rnd.NextDouble() + minX;
                velocity[k] = ((0.1 * maxX) - (0.1 * minX)) * rnd.NextDouble() + (0.1 * minX);
            }
            error = double.MaxValue; // tmp dummy value
            bestError = error;
        }
    } // Particle
    private class Swarm
    {
        // an array of particles + swarm best error
        public Particle[] particles;
        public double[] bestPosition; // of any particle
        public double bestError;

        public Swarm(int numParticles, int dim)
        {
            particles = new Particle[numParticles];
            for (int i = 0; i < numParticles; ++i)
                particles[i] = new Particle(dim);

            bestPosition = new double[dim];
            bestError = double.MaxValue;
        } // ctor
    } // Swarm

    private class MultiSwarm
    {
        // an array of Swarms + global best error
        public Swarm[] swarms;
        public double[] bestPosition; // of any particle in any swarm
        public double bestError;

        public MultiSwarm(int numSwarms, int numParticles, int dim)
        {
            swarms = new Swarm[numSwarms];
            for (int i = 0; i < numSwarms; ++i)
                swarms[i] = new Swarm(numParticles, dim);
            bestPosition = new double[dim];
            bestError = double.MaxValue;
        } // ctor
    } // MultiSwarm
    // ===== end nested classes =====
} // NeuralNetwork
