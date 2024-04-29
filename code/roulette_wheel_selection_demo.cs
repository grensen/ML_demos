
class RWS24
{
    static void Main()
    {
        Console.WriteLine("\nRoulette wheel selection demo\n");

        // keep order from high to low!
        float[] trueMachines = { 0.6f, 0.5f, 0.3f, 0.2f }; // Win probabilities of machines    
        int trials = 10;
        int RUNS = 3;
        int seed = 840635;
        int machines = trueMachines.Length;
        float impact = 0.75f;
        int topK = machines;
        int warumup = 1;

        // Hyperparameters
        Console.WriteLine($"SEED     = {seed}");
        Console.WriteLine($"MACHINES = {machines}");
        Console.WriteLine($"IMPACT   = {impact}");
        Console.WriteLine($"WARMUP   = {warumup} x all");

        // show settings
        Console.WriteLine($"\nTrue machine probabilities:");
        Console.WriteLine($"ID      Machine");
        for (int i = 0; i < trueMachines.Length; i++)
            Console.WriteLine($"{i}\t{trueMachines[i]}");
       
        Console.WriteLine($"\nDesired topK({topK}) order: {string.Join(", ", Enumerable.Range(0, topK))}");
        Console.WriteLine($"Epochs = {RUNS}, trials = {trials}");

        // default test
        Console.WriteLine("\nMachine  Expectation  Selection  Count");
        TestRWS();

        // impact test 
        for (int impct = 0; impct < 2; impct++)
        {
            impact = impct == 0? 0.2f : 0.1f;
            topK = 2;
            Console.WriteLine($"Impact = {impact:F2}, topK({topK}) order: {string.Join(", ", Enumerable.Range(0, topK))}");
            TestRWS();
        }

        void TestRWS()
        {
            Random randomMain = new Random(seed);

            float sumDistance = 0;
            for (int runs = 0; runs < RUNS; runs++)
            {
                // init selection weights + counts
                float[] predictedMachines = new float[trueMachines.Length];
                int[] rouletteSelectionCounts = new int[trueMachines.Length];
                trials = !true ? trials : (int)Math.Pow(10, runs + 1); // Number of trials for selection methods

                // init expectation
                for (int i = 0; i < trueMachines.Length; i++)
                    predictedMachines[i] = machines; // 1 / (float)(machines * 1.0f); //trials / (float)(machines * 100.0f);

                // Perform trials
                for (int i = 0; i < trials; i++)
                {
                    // select id
                    Random random = new(randomMain.Next());
                    int selectedMachine = RouletteWheelNew(predictedMachines, rouletteSelectionCounts, random, i, warumup);
                    rouletteSelectionCounts[selectedMachine]++;

                    // evaluate id
                    float rand = random.NextSingle();
                    float dist = trueMachines[selectedMachine] - rand;
                    predictedMachines[selectedMachine] +=
                        dist * dist * dist * predictedMachines[selectedMachine] * impact;
                }

                float distance = PrintTopMachines(predictedMachines, rouletteSelectionCounts, trials, topK);
                sumDistance += distance;
                // Display results for both methods
            }
            // pseudo error measurement
            Console.WriteLine($"Total error distance: {sumDistance:F2}\n");
        }

        Console.WriteLine("End demo");
    }

    static int RouletteWheelNew(float[] vals, int[] counts, Random rnd, int id, int warumup = 0)
    {
        // roulette wheel selection
        // on the fly technique
        // vals[] can't be all 0.0s
        int n = vals.Length;
        if (id != -1 && id < n * warumup) // warmup
        {
            return (id + 1) % vals.Length;
        }
        else
        {
            float sum = 0.0f;
            for (int i = 0; i < n; ++i)
                sum += vals[i] - (counts[i] / (id + 1.0f)) / n;

            float cumP = 0.0f;  // cumulative prob
            float p = rnd.NextSingle();

            float frac = 1 / sum;
            for (int i = 0; i < n; ++i)
            {
                cumP += (vals[i] * frac);
                if (cumP > p) return i;
            }
        }
        // last index
        return n - 1;  // last index
    }

    // Print top machines and their selection probabilities
    static float PrintTopMachines(float[] machineProbabilities, int[] selectionCounts, int trials, int topK)
    {
        // Find the topK machines with the highest selection counts
        int[] topIndices = new int[topK];
        int[] topCounts = new int[topK];

        for (int i = 0; i < topK; i++)
            topIndices[i] = topCounts[i] = -1;

        for (int i = 0; i < selectionCounts.Length; i++)
        {
            int count = selectionCounts[i];

            for (int j = 0; j < topK; j++)
            {
                if (count > topCounts[j])
                {
                    for (int k = topK - 1; k > j; k--)
                    {
                        topIndices[k] = topIndices[k - 1];
                        topCounts[k] = topCounts[k - 1];
                    }
                    topIndices[j] = i;
                    topCounts[j] = count;
                    break;
                }
            }
        }

        float[] prob = new float[topK];
        float sum = 0;
        // float min = prob.Min() < 0 ? -prob.Min() : 0;

        for (int i = 0; i < prob.Length; i++)
        {
            int id = topIndices[i];
            if (machineProbabilities[id] < 0) machineProbabilities[id] = 0;
            //  machineProbabilities[id] += min;
            sum += machineProbabilities[id];
        }

        for (int i = 0; i < prob.Length; i++)
        {
            int id = topIndices[i];
            prob[i] = machineProbabilities[id] / sum;
        }

        Console.WriteLine($"Trials = {trials}, distance = {TotalErrorDist(topIndices):F2}");

        // Print the top machines and their selection probabilities
        for (int i = 0; i < topK; i++)
        {
            int index = topIndices[i];
            float selectionProbability = topCounts[i] / (float)trials;
            Console.WriteLine(
                $"{index}" +
                $"\t {prob[i]:F2}\t      " +
                $"{selectionProbability:F2}\t " +
                $"{topCounts[i]}");
        }

        return TotalErrorDist(topIndices);
    }
    static float TotalErrorDist(int[] route)
    {
        float d = 0.0f;
        int n = route.Length;
        for (int i = 0; i < n; i++)
            if(route[i] != i) 
                d += Math.Abs(i - route[i]); // not final?
        return d; // / n;
    }
} // rws class





