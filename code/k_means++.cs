// https://visualstudiomagazine.com/articles/2020/05/06/data-clustering-k-means.aspx
// https://jamesmccaffrey.wordpress.com/2020/05/08/data-clustering-with-k-means-using-c/

Console.WriteLine("\nBegin k-means++ demo\n");

double[][] data = new double[20][];
data[0] = new double[] { 0.65, 0.220 };
data[1] = new double[] { 0.73, 0.160 };
data[2] = new double[] { 0.59, 0.110 };
data[3] = new double[] { 0.61, 0.120 };
data[4] = new double[] { 0.75, 0.150 };
data[5] = new double[] { 0.67, 0.240 };
data[6] = new double[] { 0.68, 0.230 };
data[7] = new double[] { 0.70, 0.220 };
data[8] = new double[] { 0.62, 0.130 };
data[9] = new double[] { 0.66, 0.210 };
data[10] = new double[] { 0.77, 0.19 };
data[11] = new double[] { 0.75, 0.180 };
data[12] = new double[] { 0.74, 0.170 };
data[13] = new double[] { 0.70, 0.210 };
data[14] = new double[] { 0.61, 0.110 };
data[15] = new double[] { 0.58, 0.100 };
data[16] = new double[] { 0.66, 0.230 };
data[17] = new double[] { 0.59, 0.120 };
data[18] = new double[] { 0.68, 0.210 };
data[19] = new double[] { 0.61, 0.130 };

int k = 3;  // number clusters
string initMethod = "plusplus";
int maxIter = 100;  // max (likely less)
int seed = 0;
Console.WriteLine($"Data = {data.Length} (x, y)");
Console.WriteLine("Setting k = " + k);
Console.WriteLine("Setting initMethod = " + initMethod);
Console.WriteLine("Setting maxIter to converge = " + maxIter);
Console.WriteLine("Setting seed = " + seed);
KMeans km = new (k, data, initMethod, maxIter, seed);

int trials = 10;  // attempts to find best
Console.WriteLine("\nStarting clustering w/ trials = " + trials);
km.Cluster(trials);
Console.WriteLine("Done");

Console.WriteLine("\nBest clustering found: ");
ShowVector(km.clustering, 3);

Console.WriteLine("\nCluster counts: ");
ShowVector(km.counts, 4);

Console.WriteLine("\nThe cluster means: ");
ShowMatrix(km.means, new int[] { 4, 4 }, new int[] { 8, 8 }, true);

Console.WriteLine("\nTotal within-cluster SS = " + km.wcss.ToString("F4"));

Console.WriteLine("\nClustered data: ");
ShowClustered(data, k, km.clustering, new int[] { 2, 3 }, new int[] { 8, 10 }, true);

Console.WriteLine("\nEnd demo");
Console.ReadLine();

static void ShowVector(int[] vec, int wid)
{
    int n = vec.Length;
    for (int i = 0; i < n; ++i)
        Console.Write(vec[i].ToString().PadLeft(wid));
    Console.WriteLine("");
}
static void ShowMatrix(double[][] m, int[] dec, int[] wid, bool indices)
{
    int n = m.Length;
    int iPad = n.ToString().Length + 1;

    for (int i = 0; i < n; ++i)
    {
        if (indices == true)
            Console.Write("[" +
              i.ToString().PadLeft(iPad) + "] ");
        for (int j = 0; j < m[0].Length; ++j)
        {
            double x = m[i][j];
            if (Math.Abs(x) < 1.0e-5) x = 0.0;
            Console.Write(x.ToString("F" +
              dec[j]).PadLeft(wid[j]));
        }
        Console.WriteLine("");
    }
}
static void ShowClustered(double[][] data, int K, int[] clustering, int[] decs, int[] wids, bool indices)
{
    int n = data.Length;
    int iPad = n.ToString().Length + 1;  // indices
    int numDash = 0;
    for (int i = 0; i < wids.Length; ++i)
        numDash += wids[i];
    for (int i = 0; i < numDash + iPad + 5; ++i)
        Console.Write("-");
    Console.WriteLine("");
    for (int k = 0; k < K; ++k) // display by cluster
    {
        for (int i = 0; i < n; ++i) // each data item
        {
            if (clustering[i] == k) // curr data is in cluster k
            {
                if (indices == true)
                    Console.Write("[" +
                      i.ToString().PadLeft(iPad) + "]   ");
                for (int j = 0; j < data[i].Length; ++j)
                {
                    double x = data[i][j];
                    if (x < 1.0e-5) x = 0.0;  // prevent "-0.00"
                    string s = x.ToString("F" + decs[j]);
                    Console.Write(s.PadLeft(wids[j]));
                }
                Console.WriteLine("");
            }
        }
        for (int i = 0; i < numDash + iPad + 5; ++i)
            Console.Write("-");
        Console.WriteLine("");
    }
} // ShowClustered()
public class KMeans
{
    public int K;  // number clusters (use lower k for indexing)
    public double[][] data;  // data to be clustered
    public int N;  // number data items
    public int dim;  // number values in each data item
    public string initMethod;  // "plusplus", "forgy" "random"
    public int maxIter;  // max per single clustering attempt
    public int[] clustering;  // final cluster assignments
    public double[][] means;  // final cluster means aka centroids
    public double wcss;  // final total within-cluster sum of squares (inertia??)
    public int[] counts;  // final num items in each cluster
    public Random rnd;  // for initialization

    public KMeans(int K, double[][] data, string initMethod, int maxIter, int seed)
    {
        this.K = K;
        this.data = data;  // reference copy
        this.initMethod = initMethod;
        this.maxIter = maxIter;

        this.N = data.Length;
        this.dim = data[0].Length;

        this.means = new double[K][];  // one mean per cluster
        for (int k = 0; k < K; ++k)
            this.means[k] = new double[this.dim];
        this.clustering = new int[N];  // cell val is cluster ID, index is data item
        this.counts = new int[K];  // one cell per cluster
        this.wcss = double.MaxValue;  // smaller is better

        this.rnd = new Random(seed);
    } // ctor
    public void Cluster(int trials)
    {
        for (int trial = 0; trial < trials; ++trial)
            Cluster();  // find a clustering and update bests
    }
    public void Cluster()
    {
        // init clustering[] and means[][] 
        // loop at most maxIter times
        //   update means using curr clustering
        //   update clustering using new means
        // end-loop
        // if clustering is new best, update clustering, means, counts, wcss

        int[] currClustering = new int[this.N];  // [0, 0, 0, 0, .. ]

        double[][] currMeans = new double[this.K][];
        for (int k = 0; k < this.K; ++k)
            currMeans[k] = new double[this.dim];

        if (this.initMethod == "plusplus")
            InitPlusPlus(this.data, currClustering, currMeans, this.rnd);
        else
            throw new Exception("not supported");

        bool changed;  //  result from UpdateClustering (to exit loop)
        int iter = 0;
        while (iter < this.maxIter)
        {
            UpdateMeans(currMeans, this.data, currClustering);
            changed = UpdateClustering(currClustering,
              this.data, currMeans);
            if (changed == false)
                break;  // need to stop iterating
            ++iter;
        }

        double currWCSS = ComputeWithinClusterSS(this.data,
          currMeans, currClustering);
        if (currWCSS < this.wcss)  // new best clustering found
        {
            // copy the clustering, means; compute counts; store WCSS
            for (int i = 0; i < this.N; ++i)
                this.clustering[i] = currClustering[i];

            for (int k = 0; k < this.K; ++k)
                for (int j = 0; j < this.dim; ++j)
                    this.means[k][j] = currMeans[k][j];

            this.counts = ComputeCounts(this.K, currClustering);
            this.wcss = currWCSS;
        }

    } // Cluster()
    private static void InitPlusPlus(double[][] data, int[] clustering, double[][] means, Random rnd)
    {
        //  k-means++ init using roulette wheel selection
        // clustering[] and means[][] exist
        int N = data.Length;
        int dim = data[0].Length;
        int K = means.Length;

        // select one data item index at random as 1st meaan
        int idx = rnd.Next(0, N); // [0, N)
        for (int j = 0; j < dim; ++j)
            means[0][j] = data[idx][j];

        for (int k = 1; k < K; ++k) // find each remaining mean
        {
            double[] dSquareds = new double[N]; // from each item to its closest mean

            for (int i = 0; i < N; ++i) // for each data item
            {
                // compute distances from data[i] to each existing mean (to find closest)
                double[] distances = new double[k]; // we currently have k means

                for (int ki = 0; ki < k; ++ki)
                    distances[ki] = EucDistance(data[i], means[ki]);

                int mi = ArgMin(distances);  // index of closest mean to curr item
                                             // save the associated distance-squared
                dSquareds[i] = distances[mi] * distances[mi];  // sq dist from item to its closest mean
            } // i

            // select an item far from its mean using roulette wheel
            // if an item has been used as a mean its distance will be 0
            // so it won't be selected

            int newMeanIdx = ProporSelect(dSquareds, rnd);
            for (int j = 0; j < dim; ++j)
                means[k][j] = data[newMeanIdx][j];
        } // k remaining means

        //Console.WriteLine("");
        //ShowMatrix(means, 4, 10);
        //Console.ReadLine();

        UpdateClustering(clustering, data, means);
    } // InitPlusPlus
    static int ProporSelect(double[] vals, Random rnd)
    {
        // roulette wheel selection
        // on the fly technique
        // vals[] can't be all 0.0s
        int n = vals.Length;

        double sum = 0.0;
        for (int i = 0; i < n; ++i)
            sum += vals[i];

        double cumP = 0.0;  // cumulative prob
        double p = rnd.NextDouble();

        for (int i = 0; i < n; ++i)
        {
            cumP += (vals[i] / sum);
            if (cumP > p) return i;
        }
        return n - 1;  // last index
    }
    private static int[] ComputeCounts(int K, int[] clustering)
    {
        int[] result = new int[K];
        for (int i = 0; i < clustering.Length; ++i)
        {
            int cid = clustering[i];
            ++result[cid];
        }
        return result;
    }
    private static void UpdateMeans(double[][] means, double[][] data, int[] clustering)
    {
        // compute the K means using data and clustering
        // assumes no empty clusters in clustering

        int K = means.Length;
        int N = data.Length;
        int dim = data[0].Length;

        int[] counts = ComputeCounts(K, clustering);  // needed for means

        for (int k = 0; k < K; ++k)  // make sure no empty clusters
            if (counts[k] == 0)
                throw new Exception("empty cluster passed to UpdateMeans()");

        double[][] result = new double[K][];  // new means
        for (int k = 0; k < K; ++k)
            result[k] = new double[dim];

        for (int i = 0; i < N; ++i)  // each data item
        {
            int cid = clustering[i];  // which cluster ID?
            for (int j = 0; j < dim; ++j)
                result[cid][j] += data[i][j];  // accumulate
        }

        // divide accum sums by counts to get means
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < dim; ++j)
                result[k][j] /= counts[k];

        // no 0-count clusters so update the means
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < dim; ++j)
                means[k][j] = result[k][j];
    }
    private static bool UpdateClustering(int[] clustering, double[][] data, double[][] means)
    {
        // update existing cluster clustering using data and means
        // proposed clustering would have an empty cluster: return false - no change to clustering
        // proposed clustering would be no change: return false, no change to clustering
        // proposed clustering is different and has no empty clusters: return true, clustering is changed

        int K = means.Length;
        int N = data.Length;

        int[] result = new int[N];  // proposed new clustering (cluster assignments)
        bool change = false;  // is there a change to the existing clustering?
        int[] counts = new int[K];  // check if new clustering makes an empty cluster

        for (int i = 0; i < N; ++i)  // make of copy of existing clustering
            result[i] = clustering[i];

        for (int i = 0; i < data.Length; ++i)  // each data item
        {
            double[] dists = new double[K];  // dist from curr item to each mean
            for (int k = 0; k < K; ++k)
                dists[k] = EucDistance(data[i], means[k]);

            int cid = ArgMin(dists);  // index of the smallest distance
            result[i] = cid;
            if (result[i] != clustering[i])
                change = true;  // the proposed clustering is different for at least one item
            ++counts[cid];
        }

        if (change == false)
            return false;  // no change to clustering -- clustering has converged

        for (int k = 0; k < K; ++k)
            if (counts[k] == 0)
                return false;  // no change to clustering because would have an empty cluster

        // there was a change and no empty clusters so update clustering
        for (int i = 0; i < N; ++i)
            clustering[i] = result[i];

        return true;  // successful change to clustering so keep looping
    }
    private static double EucDistance(double[] item, double[] mean)
    {
        // Euclidean distance from item to mean
        // used to determine cluster assignments
        double sum = 0.0;
        for (int j = 0; j < item.Length; ++j)
            sum += (item[j] - mean[j]) * (item[j] - mean[j]);
        return Math.Sqrt(sum);
    }
    private static int ArgMin(double[] v)
    {
        int minIdx = 0;
        double minVal = v[0];
        for (int i = 0; i < v.Length; ++i)
        {
            if (v[i] < minVal)
            {
                minVal = v[i];
                minIdx = i;
            }
        }
        return minIdx;
    }
    private static double ComputeWithinClusterSS(double[][] data, double[][] means, int[] clustering)
    {
        // compute total within-cluster sum of squared differences between 
        // cluster items and their cluster means
        // this is actually the objective function, not distance
        double sum = 0.0;
        for (int i = 0; i < data.Length; ++i)
        {
            int cid = clustering[i];  // which cluster does data[i] belong to?
            sum += SumSquared(data[i], means[cid]);
        }
        return sum;
    }
    private static double SumSquared(double[] item, double[] mean)
    {
        // squared distance between vectors
        // surprisingly, k-means minimizes this, not distance
        double sum = 0.0;
        for (int j = 0; j < item.Length; ++j)
            sum += (item[j] - mean[j]) * (item[j] - mean[j]);
        return sum;
    }
} // class KMeans