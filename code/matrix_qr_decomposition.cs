// https://jamesmccaffrey.wordpress.com/2023/08/10/matrix-qr-decomposition-from-scratch-using-csharp/

Console.WriteLine("\nBegin matrix QR decomposition");

double[][] M = MatCreate(4, 4);
M[0] = new double[] { 1.0, 2.0, 3.0, 4.0 };
M[1] = new double[] { 5.0, 6.0, 7.0, 8.0,};
M[2] = new double[] { 9.0, 10.0, 11.0, 12.0 };
M[3] = new double[] { 13.0, 14.0, 15.0, 16.0 };

Console.WriteLine("\nSource matrix M:");
MatShow(M, 2, 8);

Console.WriteLine("\nComputing MatQR() (standard)");
double[][] Q;
double[][] R;
MatQR(M, out Q, out R, true);
Console.WriteLine("Done");

Console.WriteLine("\nQ = ");
MatShow(Q, 8, 14);
Console.WriteLine("\nR = ");
MatShow(R, 8, 14);

double[][] QtimesR = MatProduct(Q, R);
Console.WriteLine("\nQ times R =");
MatShow(QtimesR, 8, 14);

Console.WriteLine("\nComputing MatDecomposeQR2() (no squared matrix)");
double[][] Q2;
double[][] R2;
MatDecomposeQR2(M, out Q2, out R2, true);
Console.WriteLine("Done");

Console.WriteLine("\nQ2 = ");
MatShow(Q2, 8, 14);
Console.WriteLine("\nR2 = ");
MatShow(R2, 8, 14);

static void MatQR(double[][] mat, 
    out double[][] q, out double[][] r, bool standardize)
{
    // QR decomposition, Householder algorithm
    int n = mat.Length;  // assumes mat is nxn
    int nCols = mat[0].Length;
    if (n != nCols)
        Console.WriteLine("must be square ");

    double[][] Q = MatIdentity(n);
    double[][] R = MatCopy(mat);
    for (int i = 0; i < n - 1; ++i)
    {
        double[][] H = MatIdentity(n);
        double[] a = new double[n - i];
        int k = 0;
        // last part of col [i]
        for (int ii = i; ii < n; ++ii)
            a[k++] = R[ii][i];

        double normA = VecNorm(a);
        if (a[0] < 0.0) normA = -normA;
        double[] v = new double[a.Length];
        for (int j = 0; j < v.Length; ++j)
            v[j] = a[j] / (a[0] + normA);
        v[0] = 1.0;

        double[][] h = MatIdentity(a.Length);
        double vvDot = VecDot(v, v);
        double[][] alpha = VecToMat(v, v.Length, 1);
        double[][] beta = VecToMat(v, 1, v.Length);
        double[][] aMultB = MatProduct(alpha, beta);

        for (int ii = 0; ii < h.Length; ++ii)
            for (int jj = 0; jj < h[0].Length; ++jj)
                h[ii][jj] -= (2.0 / vvDot) * aMultB[ii][jj];

        // copy h into lower right of H
        int d = n - h.Length;
        for (int ii = 0; ii < h.Length; ++ii)
            for (int jj = 0; jj < h[0].Length; ++jj)
                H[ii + d][jj + d] = h[ii][jj];

        Q = MatProduct(Q, H);
        R = MatProduct(H, R);
    } // i

    if (standardize == true)
    {
        // standardize so R diagonal is all positive
        double[][] D = MatCreate(n, n);
        for (int i = 0; i < n; ++i)
        {
            if (R[i][i] < 0.0) D[i][i] = -1.0;
            else D[i][i] = 1.0;
        }
        Q = MatProduct(Q, D);
        R = MatProduct(D, R);
    }

    q = Q;
    r = R;
}
static void MatDecomposeQR2(double[][] mat, 
    out double[][] q, out double[][] r, bool standardize)
{
    // QR decomposition, Householder algorithm.
    // doesn't require square matrix

    int m = mat.Length;
    int n = mat[0].Length;
    if (standardize == true && m != n)
        Console.WriteLine("Cannot standardize non-square");

    double[][] Q = MatIdentity(m);
    double[][] R = MatCopy(mat);
    int end;
    if (m == n) end = n - 1; else end = n;
    for (int i = 0; i < end; ++i)
    {
        double[][] H = MatIdentity(m);
        double[] a = new double[n - i];
        int k = 0;
        for (int ii = i; ii < n; ++ii)
            a[k++] = R[ii][i];

        double normA = VecNorm(a);
        if (a[0] < 0.0) { normA = -normA; }
        double[] v = new double[a.Length];
        for (int j = 0; j < v.Length; ++j)
            v[j] = a[j] / (a[0] + normA);
        v[0] = 1.0;

        double[][] h = MatIdentity(a.Length);
        double vvDot = VecDot(v, v);
        double[][] alpha = VecToMat(v, v.Length, 1);
        double[][] beta = VecToMat(v, 1, v.Length);
        double[][] aMultB = MatProduct(alpha, beta);

        for (int ii = 0; ii < h.Length; ++ii)
            for (int jj = 0; jj < h[0].Length; ++jj)
                h[ii][jj] -= (2.0 / vvDot) *
                  aMultB[ii][jj];

        // copy h into lower right of H
        int d = n - h.Length;
        for (int ii = 0; ii < h.Length; ++ii)
            for (int jj = 0; jj < h[0].Length; ++jj)
                H[ii + d][jj + d] = h[ii][jj];

        Q = MatProduct(Q, H);
        R = MatProduct(H, R);
    } // i

    if (standardize == true)
    {
        // standardize so R diagonal is all positive
        double[][] D = MatCreate(n, n);  // m == n
        for (int i = 0; i < n; ++i)
        {
            if (R[i][i] < 0.0) D[i][i] = -1.0;
            else D[i][i] = 1.0;
        }
        Q = MatProduct(Q, D);
        R = MatProduct(D, R);
    }

    q = Q;
    r = R;

}
// Implementations of missing functions
static double[][] MatCreate(int rows, int cols)
{
    double[][] result = new double[rows][];
    for (int i = 0; i < rows; ++i)
        result[i] = new double[cols];
    return result;
}
static double[][] MatIdentity(int n)
{
    double[][] result = MatCreate(n, n);
    for (int i = 0; i < n; ++i)
        result[i][i] = 1.0;
    return result;
}
static double[][] MatCopy(double[][] mat)
{
    int rows = mat.Length;
    int cols = mat[0].Length;
    double[][] result = MatCreate(rows, cols);
    for (int i = 0; i < rows; ++i)
        Array.Copy(mat[i], result[i], cols);
    return result;
}
static double VecNorm(double[] vector)
{
    double sum = 0.0;
    foreach (double val in vector)
        sum += val * val;
    return Math.Sqrt(sum);
}
static double VecDot(double[] vector1, double[] vector2)
{
    double sum = 0.0;
    int length = Math.Min(vector1.Length, vector2.Length);
    for (int i = 0; i < length; ++i)
        sum += vector1[i] * vector2[i];
    return sum;
}
static double[][] VecToMat(double[] vector, int rows, int cols)
{
    double[][] result = MatCreate(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            result[i][j] = vector[i] * vector[j];
    return result;
}
static double[][] MatProduct(double[][] mat1, double[][] mat2)
{
    int rows1 = mat1.Length;
    int cols1 = mat1[0].Length;
    int cols2 = mat2[0].Length;

    double[][] result = MatCreate(rows1, cols2);

    for (int i = 0; i < rows1; ++i)
        for (int j = 0; j < cols2; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < cols1; ++k)
                sum += mat1[i][k] * mat2[k][j];
            result[i][j] = sum;
        }

    return result;
}
static void MatShow(double[][] m, int dec, int wid)
{
    for (int i = 0; i < m.Length; ++i)
    {
        for (int j = 0; j < m[0].Length; ++j)
        {
            double v = m[i][j];
            if (Math.Abs(v) < 1.0e-5)
                v = 0.0;  // avoid "-0.00"
            Console.Write(v.ToString("F" + dec).PadLeft(wid));
        }
        Console.WriteLine("");
    }
}