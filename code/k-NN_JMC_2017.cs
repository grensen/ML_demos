// https://learn.microsoft.com/en-us/archive/msdn-magazine/2017/december/test-run-understanding-k-nn-classification-using-csharp
// code downlload: https://learn.microsoft.com/en-us/archive/msdn-magazine/2017/december/code-downloads-for-december-2017-msdn-magazine
// https://jamesmccaffrey.wordpress.com/2017/12/06/understanding-k-nn-classification-using-c/

Console.WriteLine("\nBegin k-NN classification demo\n");

bool renormalizeData = false;  // Set to true if data needs to be renormalized

double[][] trainData = LoadData();
if (renormalizeData)
    NormalizeData(trainData);

int numClasses = 3;   // 0, 1, 2

double[] unknown = new double[] { 5.25, 1.75 };
Console.WriteLine("Classifying item with predictor values: 5.25 1.75 \n");

int k = 1;
Console.WriteLine("With k = 1");
int predicted = Classify(unknown, trainData, numClasses, k);
Console.WriteLine("\nPredicted class = " + predicted);
Console.WriteLine("");

k = 4;
Console.WriteLine("With k = 4");
predicted = Classify(unknown, trainData, numClasses, k);
Console.WriteLine("\nPredicted class = " + predicted);
Console.WriteLine("");

Console.WriteLine("End k-NN demo \n");
Console.ReadLine();

static int Classify(double[] unknown, double[][] trainData, int numClasses, int k)
{
    // compute and store distances from unknown to all train data 
    int n = trainData.Length;  // number data items
    (int Index, double Distance)[] info = new (int, double)[n];
    for (int i = 0; i < n; ++i)
    {
        double dist = Distance(unknown, trainData[i]);
        info[i] = (i, dist);
    }

    Array.Sort(info, (a, b) => a.Distance.CompareTo(b.Distance));  // sort by distance
    Console.WriteLine("\nNearest  /  Distance  / Class");
    Console.WriteLine("==============================");
    for (int i = 0; i < k; ++i)
    {
        int c = (int)trainData[info[i].Index][2];
        string dist = info[i].Distance.ToString("F3");
        Console.WriteLine("( " + trainData[info[i].Index][0] + "," + trainData[info[i].Index][1] + " )  :  " + dist + "        " + c);
    }

    int result = Vote(info, trainData, numClasses, k);  // k nearest classes
    return result;

} // Classify
static int Vote((int Index, double Distance)[] info, double[][] trainData, int numClasses, int k)
{
    int[] votes = new int[numClasses];  // one cell per class
    for (int i = 0; i < k; ++i)  // just first k nearest
    {
        int idx = info[i].Index;  // which item
        int c = (int)trainData[idx][2];  // class in last cell
        ++votes[c];
    }

    int mostVotes = 0;
    int classWithMostVotes = 0;
    for (int j = 0; j < numClasses; ++j)
    {
        if (votes[j] > mostVotes)
        {
            mostVotes = votes[j];
            classWithMostVotes = j;
        }
    }

    return classWithMostVotes;
}
static double Distance(Span<double> unknown, Span<double> data)
{
    double sum = 0.0;
    for (int i = 0; i < unknown.Length; ++i)
    {
        var sqrt = unknown[i] - data[i];
        sum += sqrt * sqrt;
    }
    return Math.Sqrt(sum);
}
static double[][] LoadData()
{
    double[][] data = new double[33][];
    data[0] = new double[] { 2.0, 4.0, 0 };
    data[1] = new double[] { 2.0, 5.0, 0 };
    data[2] = new double[] { 2.0, 6.0, 0 };
    data[3] = new double[] { 3.0, 3.0, 0 };
    data[4] = new double[] { 3.0, 7.0, 0 };
    data[5] = new double[] { 4.0, 3.0, 0 };
    data[6] = new double[] { 4.0, 7.0, 0 };
    data[7] = new double[] { 5.0, 3.0, 0 };
    data[8] = new double[] { 5.0, 7.0, 0 };
    data[9] = new double[] { 6.0, 4.0, 0 };
    data[10] = new double[] { 6.0, 5.0, 0 };
    data[11] = new double[] { 6.0, 6.0, 0 };

    data[12] = new double[] { 3.0, 4.0, 1 };
    data[13] = new double[] { 3.0, 5.0, 1 };
    data[14] = new double[] { 3.0, 6.0, 1 };
    data[15] = new double[] { 4.0, 4.0, 1 };
    data[16] = new double[] { 4.0, 5.0, 1 };
    data[17] = new double[] { 4.0, 6.0, 1 };
    data[18] = new double[] { 5.0, 4.0, 1 };
    data[19] = new double[] { 5.0, 5.0, 1 };
    data[20] = new double[] { 5.0, 6.0, 1 };
    data[21] = new double[] { 6.0, 1.0, 1 };
    data[22] = new double[] { 7.0, 2.0, 1 };
    data[23] = new double[] { 8.0, 3.0, 1 };
    data[24] = new double[] { 8.0, 2.0, 1 };
    data[25] = new double[] { 8.0, 1.0, 1 };
    data[26] = new double[] { 7.0, 1.0, 1 };

    data[27] = new double[] { 2.0, 1.0, 2 };
    data[28] = new double[] { 2.0, 2.0, 2 };
    data[29] = new double[] { 3.0, 1.0, 2 };
    data[30] = new double[] { 3.0, 2.0, 2 };
    data[31] = new double[] { 4.0, 1.0, 2 };
    data[32] = new double[] { 4.0, 2.0, 2 };

    return data;
}
static void NormalizeData(double[][] data)
{
    int rows = data.Length;
    int cols = data[0].Length;

    for (int col = 0; col < cols - 1; col++) // Exclude the last column (class labels)
    {
        double min = data[0][col];
        double max = data[0][col];

        for (int row = 1; row < rows; row++)
        {
            if (data[row][col] < min)
                min = data[row][col];
            if (data[row][col] > max)
                max = data[row][col];
        }

        double range = max - min;

        for (int row = 0; row < rows; row++)
        {
            data[row][col] = (data[row][col] - min) / range;
        }
    }
}


/*
using System;
namespace KNN
{
    class KNNProgram
    {
        static void Main(string[] args)
        {
            Console.WriteLine("\nBegin k-NN classification demo\n");

            double[][] trainData = LoadData();  // get the normalized data

            int numClasses = 3;   // 0, 1, 2

            double[] unknown = new double[] { 5.25, 1.75 };
            Console.WriteLine("Classifying item with predictor values: 5.25 1.75 \n");

            int k = 1;
            Console.WriteLine("With k = 1");
            int predicted = Classify(unknown, trainData, numClasses, k);
            Console.WriteLine("\nPredicted class = " + predicted);
            Console.WriteLine("");

            k = 4;
            Console.WriteLine("With k = 4");
            predicted = Classify(unknown, trainData, numClasses, k);
            Console.WriteLine("\nPredicted class = " + predicted);
            Console.WriteLine("");

            Console.WriteLine("End k-NN demo \n");
            Console.ReadLine();
        } // Main

        static int Classify(double[] unknown, double[][] trainData, int numClasses, int k)
        {
            // compute and store distances from unknown to all train data 
            int n = trainData.Length;  // number data items
            IndexAndDistance[] info = new IndexAndDistance[n];
            for (int i = 0; i < n; ++i)
            {
                IndexAndDistance curr = new IndexAndDistance();
                double dist = Distance(unknown, trainData[i]);
                curr.idx = i;
                curr.dist = dist;
                info[i] = curr;
            }

            Array.Sort(info);  // sort by distance
            Console.WriteLine("\nNearest  /  Distance  / Class");
            Console.WriteLine("==============================");
            for (int i = 0; i < k; ++i)
            {
                int c = (int)trainData[info[i].idx][2];
                string dist = info[i].dist.ToString("F3");
                Console.WriteLine("( " + trainData[info[i].idx][0] + "," + trainData[info[i].idx][1] + " )  :  " + dist + "        " + c);
            }

            int result = Vote(info, trainData, numClasses, k);  // k nearest classes
            return result;

        } // Classify

        static int Vote(IndexAndDistance[] info, double[][] trainData, int numClasses, int k)
        {
            int[] votes = new int[numClasses];  // one cell per class
            for (int i = 0; i < k; ++i)  // just first k nearest
            {
                int idx = info[i].idx;  // which item
                int c = (int)trainData[idx][2];  // class in last cell
                ++votes[c];
            }

            int mostVotes = 0;
            int classWithMostVotes = 0;
            for (int j = 0; j < numClasses; ++j)
            {
                if (votes[j] > mostVotes)
                {
                    mostVotes = votes[j];
                    classWithMostVotes = j;
                }
            }

            return classWithMostVotes;
        }

        static double Distance(Span<double> unknown, Span<double> data)
        {
            double sum = 0.0;
            for (int i = 0; i < unknown.Length; ++i)
                sum += (unknown[i] - data[i]) * (unknown[i] - data[i]);
            return Math.Sqrt(sum);
        }

        static double[][] LoadData()
        {
            double[][] data = new double[33][];
            data[0] = new double[] { 2.0, 4.0, 0 };
            data[1] = new double[] { 2.0, 5.0, 0 };
            data[2] = new double[] { 2.0, 6.0, 0 };
            data[3] = new double[] { 3.0, 3.0, 0 };
            data[4] = new double[] { 3.0, 7.0, 0 };
            data[5] = new double[] { 4.0, 3.0, 0 };
            data[6] = new double[] { 4.0, 7.0, 0 };
            data[7] = new double[] { 5.0, 3.0, 0 };
            data[8] = new double[] { 5.0, 7.0, 0 };
            data[9] = new double[] { 6.0, 4.0, 0 };
            data[10] = new double[] { 6.0, 5.0, 0 };
            data[11] = new double[] { 6.0, 6.0, 0 };

            data[12] = new double[] { 3.0, 4.0, 1 };
            data[13] = new double[] { 3.0, 5.0, 1 };
            data[14] = new double[] { 3.0, 6.0, 1 };
            data[15] = new double[] { 4.0, 4.0, 1 };
            data[16] = new double[] { 4.0, 5.0, 1 };
            data[17] = new double[] { 4.0, 6.0, 1 };
            data[18] = new double[] { 5.0, 4.0, 1 };
            data[19] = new double[] { 5.0, 5.0, 1 };
            data[20] = new double[] { 5.0, 6.0, 1 };
            data[21] = new double[] { 6.0, 1.0, 1 };
            data[22] = new double[] { 7.0, 2.0, 1 };
            data[23] = new double[] { 8.0, 3.0, 1 };
            data[24] = new double[] { 8.0, 2.0, 1 };
            data[25] = new double[] { 8.0, 1.0, 1 };
            data[26] = new double[] { 7.0, 1.0, 1 };

            data[27] = new double[] { 2.0, 1.0, 2 };
            data[28] = new double[] { 2.0, 2.0, 2 };
            data[29] = new double[] { 3.0, 1.0, 2 };
            data[30] = new double[] { 3.0, 2.0, 2 };
            data[31] = new double[] { 4.0, 1.0, 2 };
            data[32] = new double[] { 4.0, 2.0, 2 };

            return data;
        }

    } // Program

    public class IndexAndDistance : IComparable<IndexAndDistance>
    {
        public int idx;  // index of a training item
        public double dist;  // distance from train item to unknown

        // need to sort these to find k closest
        public int CompareTo(IndexAndDistance other)
        {
            if (this.dist < other.dist) return -1;
            else if (this.dist > other.dist) return +1;
            else return 0;
        }
    }

} // ns

*/
