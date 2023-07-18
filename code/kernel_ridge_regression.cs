// https://jamesmccaffrey.wordpress.com/2023/07/17/kernel-ridge-regression-using-c-oop-style/

Console.WriteLine("\nBegin KRR using C# OOP version");

Console.WriteLine("\nLoading train and test data ");
// 1a. load training data
int[] cols = { 0, 1, 2, 3, 4, 5 };
double[][] trainX = Utils.MatToExtract(FeedTrainData(), cols[0..5]);
double[] trainY = Utils.MatToVec(Utils.MatToExtract(FeedTrainData(), cols[5..]));
// 1b. load test data
double[][] testX = Utils.MatToExtract(FeedTestData(), cols[0..5]);
double[] testY = Utils.MatToVec(Utils.MatToExtract(FeedTestData(), cols[5..]));
Console.WriteLine("Done ");

Console.WriteLine("\nFirst four X predictors: ");
for (int i = 0; i < 4; ++i) 
    Utils.VecShow(trainX[i], 4, 9, true);
Console.WriteLine("\nFirst four target y: ");
for (int i = 0; i < 4; ++i)
    Console.WriteLine(trainY[i].ToString("F4").PadLeft(8));

Console.WriteLine("\nCreating KRR object");
double gamma = 0.1;    // RBF param
double alpha = 0.001;  // regularization
Console.WriteLine("Setting gamma = " +
    gamma.ToString("F1") + ", alpha = " + alpha.ToString("F3"));
KRR krr = new KRR(gamma, alpha);
Console.WriteLine("Done ");

Console.WriteLine("\nTraining model ");
krr.Train(trainX, trainY);
Console.WriteLine("Done ");

Console.WriteLine("\nModel accuracy (within 0.10) ");
double trainAcc = Accuracy(krr, trainX, trainY, 0.10);
Console.WriteLine("\nTrain acc = " + trainAcc.ToString("F4"));
double testAcc = Accuracy(krr, testX, testY, 0.10);
Console.WriteLine("Test acc = " + testAcc.ToString("F4"));

Console.WriteLine("\nPredicting x = (0.5, -0.5, 0.5, -0.5, 0.5) ");
double[] x = new double[] { 0.5, -0.5, 0.5, -0.5, 0.5 };
double y = krr.Predict(x);
Console.WriteLine("Predicted y = " + y.ToString("F4"));

Console.WriteLine("\nEnd KRR demo ");
Console.ReadLine();

static double Accuracy(KRR model, double[][] dataX, double[] dataY, double pctClose)
{
    int numCorrect = 0; int numWrong = 0;
    int n = dataX.Length;
    for (int i = 0; i < n; ++i)
    {
        double[] x = dataX[i];
        double actualY = dataY[i];
        double predY = model.Predict(x);
        if (Math.Abs(actualY - predY) <
  Math.Abs(actualY * pctClose))
            ++numCorrect;
        else
            ++numWrong;
    }
    return (numCorrect * 1.0) / n;
}
double[][] FeedTrainData()
{
    double[][] trainData = new double[][]
    {
        new double[] { -0.1660,0.4406,-0.9998,-0.3953,-0.7065,0.4840 },  // 1
        new double[] { 0.0776,-0.1616,0.3704,-0.5911,0.7562,0.1568 },  // 2
        new double[] { -0.9452,0.3409,-0.1654,0.1174,-0.7192,0.8054 },  // 3
        new double[] { 0.9365,-0.3732,0.3846,0.7528,0.7892,0.1345 },  // 4
        new double[] { -0.8299,-0.9219,-0.6603,0.7563,-0.8033,0.7955 },  // 5
        new double[] { 0.0663,0.3838,-0.3690,0.3730,0.6693,0.3206 },  // 6
        new double[] { -0.9634,0.5003,0.9777,0.4963,-0.4391,0.7377 },  // 7
        new double[] { -0.1042,0.8172,-0.4128,-0.4244,-0.7399,0.4801 },  // 8
        new double[] { -0.9613,0.3577,-0.5767,-0.4689,-0.0169,0.6861 },  // 9
        new double[] { -0.7065,0.1786,0.3995,-0.7953,-0.1719,0.5569 },  // 10
        new double[] { 0.3888,-0.1716,-0.9001,0.0718,0.3276,0.2500 },  // 11
        new double[] { 0.1731,0.8068,-0.7251,-0.7214,0.6148,0.3297 },  // 12
        new double[] { -0.2046,-0.6693,0.8550,-0.3045,0.5016,0.2129 },  // 13
        new double[] { 0.2473,0.5019,-0.3022,-0.4601,0.7918,0.2613 },  // 14
        new double[] { -0.1438,0.9297,0.3269,0.2434,-0.7705,0.5171 },  // 15
        new double[] { 0.1568,-0.1837,-0.5259,0.8068,0.1474,0.3307 },  // 16
        new double[] { -0.9943,0.2343,-0.3467,0.0541,0.7719,0.5581 },  // 17
        new double[] { 0.2467,-0.9684,0.8589,0.3818,0.9946,0.1092 },  // 18
        new double[] { -0.6553,-0.7257,0.8652,0.3936,-0.8680,0.7018 },  // 19
        new double[] { 0.8460,0.4230,-0.7515,-0.9602,-0.9476,0.1996 },  // 20
        new double[] { -0.9434,-0.5076,0.7201,0.0777,0.1056,0.5664 },  // 21
        new double[] { 0.9392,0.1221,-0.9627,0.6013,-0.5341,0.1533 },  // 22
        new double[] { 0.6142,-0.2243,0.7271,0.4942,0.1125,0.1661 },  // 23
        new double[] { 0.4260,0.1194,-0.9749,-0.8561,0.9346,0.2230 },  // 24
        new double[] { 0.1362,-0.5934,-0.4953,0.4877,-0.6091,0.3810 },  // 25
        new double[] { 0.6937,-0.5203,-0.0125,0.2399,0.6580,0.1460 },  // 26
        new double[] { -0.6864,-0.9628,-0.8600,-0.0273,0.2127,0.5387 },  // 27
        new double[] { 0.9772,0.1595,-0.2397,0.1019,0.4907,0.1611 },  // 28
        new double[] { 0.3385,-0.4702,-0.8673,-0.2598,0.2594,0.2270 },  // 29
        new double[] { -0.8669,-0.4794,0.6095,-0.6131,0.2789,0.4700 },  // 30
        new double[] { 0.0493,0.8496,-0.4734,-0.8681,0.4701,0.3516 },  // 31
        new double[] { 0.8639,-0.9721,-0.5313,0.2336,0.8980,0.1412 },  // 32
        new double[] { 0.9004,0.1133,0.8312,0.2831,-0.2200,0.1782 },  // 33
        new double[] { 0.0991,0.8524,0.8375,-0.2102,0.9265,0.2150 },  // 34
        new double[] { -0.6521,-0.7473,-0.7298,0.0113,-0.9570,0.7422 },  // 35
        new double[] { 0.6190,-0.3105,0.8802,0.1640,0.7577,0.1056 },  // 36
        new double[] { 0.6895,0.8108,-0.0802,0.0927,0.5972,0.2214 },  // 37
        new double[] { 0.1982,-0.9689,0.1870,-0.1326,0.6147,0.1310 },  // 38
        new double[] { -0.3695,0.7858,0.1557,-0.6320,0.5759,0.3773 },  // 39
        new double[] { -0.1596,0.3581,0.8372,-0.9992,0.9535,0.2071 },  // 40
    };
    return trainData;
}
double[][] FeedTestData()
{
    double[][] testData = new double[][]
    {
        new double[] {-0.2468,0.9476,0.2094,0.6577,0.1494,0.4132 },  // 1
        new double[] {0.1737,0.5000,0.7166,0.5102,0.3961,0.2611 },  // 2
        new double[] {0.7290,-0.3546,0.3416,-0.0983,-0.2358,0.1332 },  // 3
        new double[] {-0.3652,0.2438,-0.1395,0.9476,0.3556,0.4170 },  // 4
        new double[] {-0.6029,-0.1466,-0.3133,0.5953,0.7600,0.4334 },  // 5
        new double[] {-0.4596,-0.4953,0.7098,0.0554,0.6043,0.2775 },  // 6
        new double[] {0.1450,0.4663,0.0380,0.5418,0.1377,0.2931 },  // 7
        new double[] {-0.8636,-0.2442,-0.8407,0.9656,-0.6368,0.7429 },  // 8
        new double[] {0.6237,0.7499,0.3768,0.1390,-0.6781,0.2185 },  // 9
        new double[] {-0.5499,0.1850,-0.3755,0.8326,0.8193,0.4399 },  // 10
    };
    return testData;
}
public class KRR
{
    public double gamma;
    public double alpha;
    public double[][] trainX;
    public double[] wts;

    public KRR(double gamma, double alpha)
    {
        this.gamma = gamma;
        this.alpha = alpha;
    } // ctor

    public void Train(double[][] trainX, double[] trainY)
    {
        // 0. store trainX (needed by Predict()
        this.trainX = trainX;  // by ref -- could copy

        // 1. compute K matrix
        int N = trainX.Length;
        double[][] K = Utils.MatCreate(N, N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                K[i][j] = this.Rbf(trainX[i], trainX[j]);

        // 2. add regularization on diagonal
        for (int i = 0; i < N; ++i)
            K[i][i] += this.alpha;

        // 3. compute Kinv
        double[][] Kinv = Utils.MatInverse(K);

        // 4. compute model weights using K inverse
        this.wts = Utils.VecMatProd(trainY, Kinv);

    } // Train

    public double Rbf(double[] v1, double[] v2)
    {
        int dim = v1.Length;
        double sum = 0.0;
        for (int i = 0; i < dim; ++i)
        {
            sum += (v1[i] - v2[i]) * (v1[i] - v2[i]);
        }
        return Math.Exp(-1 * this.gamma * sum);
    }

    public double Predict(double[] x)
    {
        int N = this.trainX.Length;
        double sum = 0.0;
        for (int i = 0; i < N; ++i)
        {
            double[] xx = this.trainX[i];
            double k = this.Rbf(x, xx);
            sum += this.wts[i] * k;
        }
        return sum;
    }




} // class KRR
public class Utils
{
    //public static double[][] VecToMat(double[] vec)
    //{
    //  // vector to row vec/matrix
    //  double[][] result = MatCreate(vec.Length, 1);
    //  for (int i = 0; i < vec.Length; ++i)
    //    result[i][0] = vec[i];
    //  return result;
    //}
    public static double[][] MatCreate(int rows, int cols)
    {
        double[][] result = new double[rows][];
        for (int i = 0; i < rows; ++i)
            result[i] = new double[cols];
        return result;
    }

    public static double[] MatToVec(double[][] m)
    {
        int rows = m.Length;
        int cols = m[0].Length;
        double[] result = new double[rows * cols];
        int k = 0;
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
            {
                result[k++] = m[i][j];
            }
        return result;
    }

    public static double[][] MatProduct(double[][] matA,
      double[][] matB)
    {
        int aRows = matA.Length;
        int aCols = matA[0].Length;
        int bRows = matB.Length;
        int bCols = matB[0].Length;
        if (aCols != bRows)
            throw new Exception("Non-conformable matrices");

        double[][] result = MatCreate(aRows, bCols);

        for (int i = 0; i < aRows; ++i) // each row of A
            for (int j = 0; j < bCols; ++j) // each col of B
                for (int k = 0; k < aCols; ++k)
                    result[i][j] += matA[i][k] * matB[k][j];

        return result;
    }

    public static double[] VecMatProd(double[] v,
      double[][] m)
    {
        // ex: 
        int nRows = m.Length;
        int nCols = m[0].Length;
        int n = v.Length;
        if (n != nCols)
            throw new Exception("non-comform in VecMatProd");

        double[] result = new double[n];
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < nCols; ++j)
            {
                result[i] += v[j] * m[i][j];
            }
        }
        return result;
    }

    public static double[] MatVecProd(double[][] mat,
      double[] vec)
    {
        int nRows = mat.Length; int nCols = mat[0].Length;
        if (vec.Length != nCols)
            throw new Exception("Non-conforme in MatVecProd() ");
        double[] result = new double[nRows];
        for (int i = 0; i < nRows; ++i)
        {
            for (int j = 0; j < nCols; ++j)
            {
                result[i] += mat[i][j] * vec[j];
            }
        }
        return result;
    }

    public static double[][] MatTranspose(double[][] m)
    {
        int nr = m.Length;
        int nc = m[0].Length;
        double[][] result = MatCreate(nc, nr);  // note
        for (int i = 0; i < nr; ++i)
            for (int j = 0; j < nc; ++j)
                result[j][i] = m[i][j];
        return result;
    }

    // -------

    public static double[][] MatInverse(double[][] m)
    {
        // assumes determinant is not 0
        // that is, the matrix does have an inverse
        int n = m.Length;
        double[][] result = MatCreate(n, n); // make a copy
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                result[i][j] = m[i][j];

        double[][] lum; // combined lower & upper
        int[] perm;  // out parameter
        MatDecompose(m, out lum, out perm);  // ignore return

        double[] b = new double[n];
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
                if (i == perm[j])
                    b[j] = 1.0;
                else
                    b[j] = 0.0;

            double[] x = Reduce(lum, b); // 
            for (int j = 0; j < n; ++j)
                result[j][i] = x[j];
        }
        return result;
    }

    static int MatDecompose(double[][] m,
      out double[][] lum, out int[] perm)
    {
        // Crout's LU decomposition for matrix determinant
        // and inverse.
        // stores combined lower & upper in lum[][]
        // stores row permuations into perm[]
        // returns +1 or -1 according to even or odd number
        // of row permutations.
        // lower gets dummy 1.0s on diagonal (0.0s above)
        // upper gets lum values on diagonal (0.0s below)

        // even (+1) or odd (-1) row permutatuions
        int toggle = +1;
        int n = m.Length;

        // make a copy of m[][] into result lu[][]
        lum = MatCreate(n, n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                lum[i][j] = m[i][j];

        // make perm[]
        perm = new int[n];
        for (int i = 0; i < n; ++i)
            perm[i] = i;

        for (int j = 0; j < n - 1; ++j) // note n-1 
        {
            double max = Math.Abs(lum[j][j]);
            int piv = j;

            for (int i = j + 1; i < n; ++i) // pivot index
            {
                double xij = Math.Abs(lum[i][j]);
                if (xij > max)
                {
                    max = xij;
                    piv = i;
                }
            } // i

            if (piv != j)
            {
                double[] tmp = lum[piv]; // swap rows j, piv
                lum[piv] = lum[j];
                lum[j] = tmp;

                int t = perm[piv]; // swap perm elements
                perm[piv] = perm[j];
                perm[j] = t;

                toggle = -toggle;
            }

            double xjj = lum[j][j];
            if (xjj != 0.0)
            {
                for (int i = j + 1; i < n; ++i)
                {
                    double xij = lum[i][j] / xjj;
                    lum[i][j] = xij;
                    for (int k = j + 1; k < n; ++k)
                        lum[i][k] -= xij * lum[j][k];
                }
            }

        } // j

        return toggle;  // for determinant
    } // MatDecompose

    static double[] Reduce(double[][] luMatrix,
      double[] b) // helper
    {
        int n = luMatrix.Length;
        double[] x = new double[n];
        //b.CopyTo(x, 0);
        for (int i = 0; i < n; ++i)
            x[i] = b[i];

        for (int i = 1; i < n; ++i)
        {
            double sum = x[i];
            for (int j = 0; j < i; ++j)
                sum -= luMatrix[i][j] * x[j];
            x[i] = sum;
        }

        x[n - 1] /= luMatrix[n - 1][n - 1];
        for (int i = n - 2; i >= 0; --i)
        {
            double sum = x[i];
            for (int j = i + 1; j < n; ++j)
                sum -= luMatrix[i][j] * x[j];
            x[i] = sum / luMatrix[i][i];
        }

        return x;
    } // Reduce
    static int NumNonCommentLines(string fn, string comment)
    {
        int ct = 0;
        string line = "";
        FileStream ifs = new FileStream(fn,
          FileMode.Open);
        StreamReader sr = new StreamReader(ifs);
        while ((line = sr.ReadLine()) != null)
            if (line.StartsWith(comment) == false)
                ++ct;
        sr.Close(); ifs.Close();
        return ct;
    }
    public static double[][] MatLoad(string fn, int[] usecols, char sep, string comment)
    {
        // count number of non-comment lines
        int nRows = NumNonCommentLines(fn, comment);
        int nCols = usecols.Length;
        double[][] result = MatCreate(nRows, nCols);
        string line = "";
        string[] tokens = null;
        FileStream ifs = new FileStream(fn, FileMode.Open);
        StreamReader sr = new StreamReader(ifs);

        int i = 0;
        while ((line = sr.ReadLine()) != null)
        {
            if (line.StartsWith(comment) == true)
                continue;
            tokens = line.Split(sep);
            for (int j = 0; j < nCols; ++j)
            {
                int k = usecols[j];  // into tokens
                result[i][j] = double.Parse(tokens[k]);
            }
            ++i;
        }
        sr.Close(); ifs.Close();
        return result;
    }
    public static double[][] MatMakeDesign(double[][] m)
    {
        // add a leading column of 1s to create a design matrix
        int nRows = m.Length; int nCols = m[0].Length;
        double[][] result = MatCreate(nRows, nCols + 1);
        for (int i = 0; i < nRows; ++i)
            result[i][0] = 1.0;

        for (int i = 0; i < nRows; ++i)
            for (int j = 0; j < nCols; ++j)
                result[i][j + 1] = m[i][j];
        return result;
    }
    public static void MatShow(double[][] m, int dec, int wid)
    {
        for (int i = 0; i < m.Length; ++i)
        {
            for (int j = 0; j < m[0].Length; ++j)
            {
                double v = m[i][j];
                if (Math.Abs(v) < 1.0e-8) v = 0.0; // hack
                Console.Write(v.ToString("F" +
                  dec).PadLeft(wid));
            }
            Console.WriteLine("");
        }
    }
    public static double[][] MatToExtract(double[][] mat, int[] cols)
    {
        int nRows = mat.Length;
        int nCols = cols.Length;
        double[][] result = MatCreate(nRows, nCols);
        for (int i = 0; i < nRows; ++i)
            for (int j = 0; j < nCols; ++j)  // idx into src cols
            {
                int srcCol = cols[j];
                int destCol = j;
                result[i][destCol] = mat[i][srcCol];
            }
        return result;
        static double[][] MatCreate(int rows, int cols)
        {
            double[][] result = new double[rows][];
            for (int i = 0; i < rows; ++i)
                result[i] = new double[cols];
            return result;
        }
    }
    public static void VecShow(double[] vec, int dec, int wid, bool newLine)
    {
        for (int i = 0; i < vec.Length; ++i)
        {
            double x = vec[i];
            if (Math.Abs(x) < 1.0e-8) x = 0.0;  // hack
            Console.Write(x.ToString("F" +
              dec).PadLeft(wid));
        }
        if (newLine == true)
            Console.WriteLine("");
    }
} // class Utils
