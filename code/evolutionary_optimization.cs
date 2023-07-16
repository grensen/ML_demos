// https://jamesmccaffrey.wordpress.com/2023/01/12/evolutionary-optimization-using-c-2/

Console.WriteLine("\nBegin evolutionary hyperparameter optimization search\n");

int numHP  = 10;
int numHPV = 6;
int numPop = 3;

Console.WriteLine($"Setting {numHP} hyperparameters, {numHPV} values ");
Console.WriteLine($"Population = {numPop}, seed = 1337");
Console.WriteLine("Optimal result is all 0s ");

Solver solver = new Solver(numHP, numHPV, numPop, 1337);
Console.WriteLine("\nInitial population:  | Error: ");
solver.Show();

Console.WriteLine("\nBegin evolutionary search ");
solver.Solve(1000);
Console.WriteLine("\nDone ");

Console.WriteLine("\nFinal population:    | Error: ");
solver.Show();

Console.WriteLine("\nEnd demo ");
Console.ReadLine();

public class Solver
{
    // parallel arrays version
    public Random rnd;
    public int numHP;  // number of hyperparameters, like 10
    public int numHPV; // number hyperparam values, like 8
    public int numPop;

    public int[][] solns;  // defines the population
    public double[] errs;

    public int[] bestSoln;
    public double bestErr;

    public Solver(int numHP, int numHPV, int numPop, int seed)
    {
        this.rnd = new Random(seed);
        this.numHP = numHP;
        this.numHPV = numHPV;
        this.numPop = numPop;

        this.solns = new int[numPop][];  // allocate
        for (int i = 0; i < numPop; ++i)
            this.solns[i] = new int[numHP];
        this.errs = new double[numPop];

        for (int i = 0; i < numPop; ++i)  // each soln
        {
            for (int j = 0; j < numHP; ++j)  // each hyperpram 
                this.solns[i][j] = this.rnd.Next(0, numHPV);
            this.errs[i] = ComputeError(this.solns[i]);
        }

        Array.Sort(this.errs, this.solns);  // sort by err

        this.bestSoln = new int[numHP];
        for (int i = 0; i < this.numHP; ++i)
            this.bestSoln[i] = this.solns[0][i];
        this.bestErr = this.errs[0];
    } // ctor
    public void Show()
    {
        for (int i = 0; i < this.numPop; ++i)
        {
            for (int j = 0; j < this.numHP; ++j)
                Console.Write(this.solns[i][j] + " ");
            Console.WriteLine(" | " + this.errs[i].ToString("F4"));
        }

        Console.WriteLine("-----");
        for (int i = 0; i < this.numHP; ++i)
            Console.Write(this.bestSoln[i] + " ");
        Console.WriteLine(" | " + this.bestErr.ToString("F4"));
    } // Show()
    public double ComputeError(int[] soln)
    {
        double err = 0.0;
        for (int i = 0; i < soln.Length; ++i)
            err += soln[i];
        return err;
    }
    public int[] PickParents() // modified from source
    {
        int half = this.numPop / 2;
        int middle = rnd.Next(half / 2, half + half / 2);

        int start = rnd.Next(0, middle);
        int first = rnd.Next(start, middle);

        int end = rnd.Next(middle, this.numPop);
        int second = rnd.Next(middle, end);

        int flip = rnd.Next(0, 2);  // 0 or 1
        if (flip == 0)
            return new int[] { first, second };
        else
            return new int[] { second, first };
    }
    public int[] CrossOver(int i, int j)
    {
        int[] child = new int[this.numHP];
        int[] parent1 = this.solns[i];
        int[] parent2 = this.solns[j];

        for (int k = 0; k < this.numHP / 2; ++k)
            child[k] = parent1[k];
        for (int k = this.numHP / 2; k < this.numHP; ++k)
            child[k] = parent2[k];
        return child;
    }
    public void Mutate(int[] soln)
    {
        int idx = rnd.Next(0, soln.Length);
        int flip = rnd.Next(0, 2);  // 0 or 1
        if (flip == 0)
        {
            soln[idx] -= 1;
            if (soln[idx] == -1)  // too small
                soln[idx] = this.numHPV - 1;
        }
        else  // flip == 1
        {
            soln[idx] += 1;
            if (soln[idx] == this.numHPV)  // too big
                soln[idx] = 0;
        }
    }
    public void Solve(int maxGen)
    {
        for (int gen = 0; gen < maxGen; ++gen)
        {
            // 1. make a child
            int[] parents = this.PickParents();  // two indices
            int[] child = this.CrossOver(parents[0], parents[1]);

            // 2. mutate and evaluate child
            this.Mutate(child);
            double childErr = this.ComputeError(child);

            // 3. is child new best solution?
            if (childErr < this.bestErr)
            {
                Console.WriteLine("New best soln found at generation " + gen);
                for (int i = 0; i < child.Length; ++i)
                    this.bestSoln[i] = child[i];
                this.bestErr = childErr;
            }

            // 4. replace a weak soln with child
            int idx = rnd.Next(this.numPop / 2, this.numPop);
            for (int j = 0; j < this.numHP; ++j)
                this.solns[idx][j] = child[j];
            this.errs[idx] = childErr;

            // 5. sort
            Array.Sort(this.errs, this.solns);
        }  // for
    } // Solve
} // Solver3 class