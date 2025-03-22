// https://github.com/grensen/easy_regression

using System.Linq;
using System.Net;

Console.WriteLine("\nBegin Easy Regression on credit card fraud demo\n");

string yourPath = @"C:\fraud_detection\";

// load credit card fraud detection dataset
var (train, test, featureNames, labelNames) = AutoDataInitCreditCard(yourPath);

int id = 0; // sample index

ShowSample("Original", train, featureNames, id);

MinMaxNormWithScaling(train, test, 0, 1, 2.5f);
ShowSample("Normalized", train, featureNames, id);

int EPOCHS = 5;
float LEARNING = 2;
Console.WriteLine($"\nUsed hyperparameters: EPOCHS = {EPOCHS}, LEARNING = {LEARNING:F1}\n");

var weights = RunEasyRegressionTraining(train, labelNames, LEARNING, EPOCHS);

ComputeAccuracy(train, labelNames, weights, "Training");
ComputeAccuracy(test, labelNames, weights, "Test");

int p = MakePrediction(train[id], labelNames.Length, weights);

Console.WriteLine($"\nPrediction for ID {id}: Class {p} {labelNames[p]}");
Console.WriteLine($"\nEnd demo");

static (float[][] train, float[][] test, string[] featureNames, string[] labelNames) AutoDataInitCreditCard(string yourPath)
{
    // Directory path to save file
    string localFilePath = Path.Combine(yourPath, "creditcardData.txt");

    // Check if directory path exists and create if not
    if (!Directory.Exists(yourPath))
        Directory.CreateDirectory(yourPath);

    // Check if data file exists, else download
    if (!File.Exists(localFilePath))
    {
        // string dataUrl = "https://datahub.io/machine-learning/creditcard/r/creditcard.csv";
        string dataUrl = "https://synapseaisolutionsa.blob.core.windows.net/public/Credit_Card_Fraud_Detection/creditcard.csv";
        Console.WriteLine($"Data not found! Download from:\n\t{dataUrl}");
        byte[] data = new HttpClient().GetByteArrayAsync(dataUrl).Result;
        File.WriteAllBytes(localFilePath, data);
    }

    string[] trainDataLines = File.ReadAllLines(localFilePath).ToArray();

    // check first lines
    // Console.WriteLine(trainDataLines[0]);
    // Console.WriteLine(trainDataLines[1]);

    Console.WriteLine("Dataset: Credit Card Fraud Detection (" + yourPath + ")");

    // Load original data into float array
    float[][] trainData = new float[trainDataLines.Length - 1][];
    for (int i = 1; i < trainDataLines.Length; i++)
    {
        string[] values = trainDataLines[i]
            .Split(',')
            .Select(x => x.Replace("\"", ""))
            .ToArray();

        float[] rowData = new float[values.Length];
        for (int j = 0; j < values.Length; j++)
            if (j == 30 && int.TryParse(values[j].Trim('\''), out int labelValue))
                rowData[j] = labelValue;
            else if (float.TryParse(values[j], out float parsedValue))
                rowData[j] = parsedValue;
        trainData[i - 1] = rowData;
    }

    Console.WriteLine($"Remove features Time, Amount and others from {trainData.Length} samples");

    // Define feature indices to keep (excluding Time and Amount)
    int[] ids = [
        // 0, remove time
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27,
        28, 29
    // 30, remove amount
    ];

    // Remove first feature (Time) and second-to-last feature (Amount), keep label at last
    float[][] trainDataMod = new float[trainData.Length][];
    for (int i = 0; i < trainData.Length; i++)
    {
        trainDataMod[i] = new float[ids.Length + 1]; // +1 for label
        for (int j = 0; j < ids.Length; j++)
        {
            int id = ids[j];
            trainDataMod[i][j] = trainData[i][id];
        }
        trainDataMod[i][^1] = trainData[i][trainData[0].Length - 1]; // Copy label
    }
    Console.WriteLine($"Reduce {trainDataMod[0].Length - 1} Features with FeatureSelector");

    // Apply feature selection

    trainDataMod = true ? new FeatureSelector().SelectBestFeatures(trainDataMod)
        : new HybridFeatureSelector().SelectBestFeatures(trainDataMod, miThreshold: 0.0001f, minFeatures: 25);

    // Split dataset into train and test
    Console.WriteLine($"Split dataset from 2 days with {trainDataMod[0].Length - 1} features and label");
    int splitIndex = trainDataMod.Length / 2;
    float[][] train = trainDataMod.Take(splitIndex).ToArray();
    float[][] test = trainDataMod.Skip(splitIndex).ToArray();

    Console.WriteLine($"Training day 1 = {train.Length} samples");
    Console.WriteLine($"Test day 2     = {test.Length} samples");

    // Extract feature names
    string[] featureNames = trainDataLines[0]
        .Split(',')
        .Skip(1)
        .Take(trainDataMod[0].Length - 1)
        .Select(f => f.Trim('"')) // format fix from "..." to '...' 
        .ToArray();

    string[] labelNames = { "Transaction", "Fraud" };

    return (train, test, featureNames, labelNames);
}

static void MinMaxNormWithScaling(float[][] train, float[][] test, float min, float max, float factor)
{
    Console.WriteLine($"\nMin-Max ({min},{max}) normalization with scaling factor {factor}");

    // Calculate min and max values for each feature in training set
    float[] minValues = new float[train[0].Length], maxValues = new float[train[0].Length];
    for (int i = 0; i < train[0].Length - 1; i++)
    {
        minValues[i] = maxValues[i] = train[0][i];
        for (int j = 1; j < train.Length; j++)
        {
            if (train[j][i] < minValues[i]) minValues[i] = train[j][i];
            if (train[j][i] > maxValues[i]) maxValues[i] = train[j][i];
        }
    }

    // Apply scaling factor
    for (int i = 0; i < train[0].Length - 1; i++)
    {
        minValues[i] *= factor;
        maxValues[i] *= factor;
    }

    Console.WriteLine($"Normalize training and test data with boundaries based on training day 1");
    NormData(train, minValues, maxValues, min, max);
    NormData(test, minValues, maxValues, min, max);

    static void NormData(float[][] data, float[] minVal, float[] maxVal, float min, float max)
    {
        for (int i = 0; i < data.Length; i++)
            for (int j = 0; j < data[i].Length - 1; j++)
                data[i][j] = ((data[i][j] - minVal[j]) / (maxVal[j] - minVal[j])) * (max - min) + min;
    }
}

static float[] RunEasyRegressionTraining(float[][] train, string[] labels, float lr, int epochs)
{
    int inputSize = train[0].Length - 1, outputSize = labels.Length;
    // init zero weights
    float[] weights = new float[inputSize * outputSize];
    Console.WriteLine($"Initialize linear network layer {inputSize}-{outputSize} with {weights.Length} zero weights");

    Random rng = new(1337);

    Console.WriteLine($"\nStart Training:");
    for (int ep = 0; ep < epochs; ep++)
    {
        // shuffle indices with Fisher-Yates 
        int[] indices = Shuffle(train.Length, rng.Next());

        for (int i = 0; i < weights.Length; i++) weights[i] *= lr;
        // train model and check accuracy
        int correct = 0;
        for (int x = 0; x < train.Length; x++)
            correct += Eval(train[indices[x]], labels.Length, weights, true);

        // info each epoch // if (ep == 0 || ep == epochs - 1 || ep == epochs / 2)
        Console.WriteLine($"Epoch = {(ep + 1),1} | Accuracy = {correct * 100.0 / train.Length:F2}%");
    }
    return weights;

    static int[] Shuffle(int length, int seed, bool doShuffle = true)
    {
        Random random = new Random(seed);
        int[] indices = new int[length];

        for (int i = 0; i < length; i++)
            indices[i] = i;

        if (!doShuffle) return indices;

        for (int i = length - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        return indices;
    }
}
static float[] ComputeAccuracy(float[][] data, string[] labels, float[] weights, string str)
{
    int inputSize = data[0].Length - 1, outputSize = labels.Length;

    int[] classCorrect = new int[outputSize], classAll = new int[outputSize];

    // test weights on given dataset
    for (int x = 0; x < data.Length; x++)
    {
        int label = (int)data[x][data[0].Length - 1];
        if (Eval(data[x], labels.Length, weights) == 1) classCorrect[label]++;
        classAll[label]++;
    }

    Console.WriteLine($"\n{str + " accuracy",-17} = {classCorrect.Sum() * 100.0 / classAll.Sum(),7:F2}%" 
        + $" ({classCorrect.Sum(),6}|{classAll.Sum() - classCorrect.Sum(),3}|{classAll.Sum(),6})");

    for (int i = 0; i < labels.Length; i++)
        Console.WriteLine($"{labels[i],-17} = {classCorrect[i] * 100.0 / classAll[i],7:F2}%"
            + $" ({classCorrect[i],6}|{classAll[i] - classCorrect[i],3}|{classAll[i],6})");

    return weights;
}
static int Eval(float[] data, int classes, float[] weights, bool training = false)
{
    int inputSize = data.Length - 1, outputSize = classes;

    int prediction = MakePrediction(data, classes, weights);

    int targetClass = (int)data[inputSize]; // class label at last

    if (training) // (backprop) plus update
        if (prediction != targetClass)
            for (int i = 0; i < inputSize; i++)
            {
                weights[i * outputSize + targetClass] += data[i];
                weights[i * outputSize + prediction] -= data[i];
            }

    // check prediction
    return prediction == targetClass ? 1 : 0;
}
static int MakePrediction(float[] data, int classes, float[] weights)
{
    int inputSize = data.Length - 1;

    // feed forward
    float[] outputs = new float[classes];
    for (int i = 0; i < inputSize; i++)
        for (int j = 0; j < classes; j++)
            outputs[j] += data[i] * weights[i * classes + j];
    
    // argmax
    int p = 0;
    for (int i = 1; i < outputs.Length; i++)
        p = outputs[i] > outputs[p] ? i : p;

    // return prediction
    return p;
}
static void ShowSample(string str, float[][] data, string[] featureNames, int id)
{
    Console.WriteLine($"\n{str} training ID {id}, class label {(int)data[id][^1]}:");

    SplitFeatures(0, (data[0].Length - 1) / 2);

    SplitFeatures((data[0].Length - 1) / 2, data[0].Length - 1);
   
    void SplitFeatures(int start, int end)
    {
        float[] dataPoint = data[id];
        Console.Write($"ID ");
        for (int i = start; i < end; i++)
            Console.Write($"{featureNames[i],-5} ");
        Console.WriteLine();

        Console.Write($"{id}  ");
        for (int i = start; i < end; i++)
            Console.Write($"{dataPoint[i],-5:F1} ");
        Console.WriteLine();
    }
}
