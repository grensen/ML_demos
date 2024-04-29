
using System.Collections.Generic;
using System.Linq;
using System;

class PermutationBlock
{
    static void Main()
    {
        Console.WriteLine($"\nPermutation block in lexicographical order demo");

        string[] filterNames = { "sharp", "flip", "blur", "shift", "drop" };
        int[] filterNameIds = { 14, 6, 9, 19, 23 };
        int permutationSize = 4; // max filter len

        // first element exclusive
        int allIds = 1;

        // Calculate and display the number of permutations for each permutation size
        for (int k = 1; k <= permutationSize; k++)
        {
            int numPermutations = Factorial(permutationSize) / Factorial(permutationSize - k);
            //Console.WriteLine($"Permutations({permutationSize}) size({k}): {numPermutations}");
            allIds += numPermutations;
        }
        Console.WriteLine($"\nPermutation block size({permutationSize}), id size: {allIds}");

        Console.WriteLine($"\nBlock raw, no first = {allIds} x {permutationSize}");

        // init -1 (-2) to disable all, then fill with permutations
        int[] permBlock = new int[allIds * permutationSize];

        // -2 to increment 1
        for (int i = 0; i < permBlock.Length; i++)
            permBlock[i] = -2;

        // raw block before first element
        CreateRawPermutationBlock(permutationSize, permBlock);

        Console.WriteLine($"\nBlock with empty positions (-2) = {allIds} x {permutationSize}");

        PrintPermutation(permBlock, permutationSize);

        // increment to final positions
        for (int i = 0; i < allIds; i++)
            for (int j = 0; j < permutationSize; ++j)
                permBlock[i * permutationSize + j]++;

        int firstID = 0;

        // add first element to permutation block
        int permSizeNew = permutationSize + 1;
        Console.WriteLine($"\nAdd first element({firstID}) and increment " +
            $"= {allIds} x {permSizeNew}");

        int[] permBlockReady = new int[allIds * permSizeNew];
        for (int i = 0; i < allIds; i++)
        {
            permBlockReady[i * permSizeNew] = firstID;
            for (int j = 1; j < permSizeNew; ++j)
                permBlockReady[i * permSizeNew + j] = permBlock[i * permutationSize + j - 1];
        }

        PrintPermutation(permBlockReady, permSizeNew);

        // Printing the result
        Console.WriteLine($"\nPossible filters:" + "\n" +
          string.Join(", ", filterNames.Zip(filterNameIds, (name, id) => $"{name}({id})")));
        // Console.WriteLine("\n"+string.Join(", ", filterNames)); Console.WriteLine("\n" + string.Join(", ", filterNameIds));

        Console.WriteLine($"\nTranslated names:");
        PrintPermutationNames(permBlockReady, filterNames, permSizeNew);

        for (int c = 0; c < permBlockReady.Length; c++)
            if (permBlockReady[c] >= 0) // drop -1 or less
                permBlockReady[c] = filterNameIds[permBlockReady[c]];

        Console.WriteLine($"\nTranslated positions:");
        PrintPermutation(permBlockReady, permSizeNew);

        Console.WriteLine($"\nEnd permutation block demo");
    }

    static void CreateRawPermutationBlock(int permutationSize, int[] permBlock)
    {
        int id = 1;
        // create one permutation block
        for (int perms = 0; perms < permutationSize; perms++)
        {
            int n = permutationSize;
            int k = perms + 1;
            int[] permutation = new int[k];

            Stack<(int[], int)> stack = new Stack<(int[], int)>();
            stack.Push((permutation, 0));

            while (stack.Count > 0)
            {
                var (currentPermutation, currentIndex) = stack.Pop();

                if (currentIndex == k)
                {
                    // grap all different permutation ids
                    for (int i = 0; i < currentPermutation.Length; i++)
                        permBlock[id * permutationSize + i] = currentPermutation[i];
                    id++; // track permutation id
                    Console.WriteLine($"{-1 + id,-3}:  {string.Join(", ", currentPermutation),-2}");
                    continue;
                }

                for (int i = n - 1; i >= 0; i--)
                    if (!IsInArray(currentPermutation, currentIndex, i))
                    {
                        currentPermutation[currentIndex] = i;
                        stack.Push((currentPermutation.ToArray(), currentIndex + 1));
                        currentPermutation[currentIndex] = -1; // Reset the value for backtracking
                    }
            }
        }
    }

    static bool IsInArray(int[] array, int length, int value)
    {
        for (int i = 0; i < length; i++)
        {
            if (array[i] == value)
            {
                return true;
            }
        }
        return false;
    }

    static int Factorial(int n)
    {
        if (n == 0) return 1;
        int result = 1;
        for (int i = 1; i <= n; i++)
            result *= i;
        return result;
    }

    static void PrintPermutationNames(int[] permBlock, string[] names, int permSize)
    {
        int permLen = permBlock.Length / permSize;
        for (int i = 0; i < permLen; i++)
        {
            Console.Write($"{i,-3}: ");
            for (int j = 0; j < permSize - 0; ++j)
                if (permBlock[i * permSize + j] >= 0)
                    Console.Write($"{names[permBlock[i * permSize + j]],5}"
                        + (permSize - 1 == j ? "" : ", "));
            Console.WriteLine($"");
        }
    }

    static void PrintPermutation(int[] permBlock, int permSize)
    {
        int permLen = permBlock.Length / permSize;
        for (int i = 0; i < permLen; i++)
        {
            Console.Write($"{i,-3}: ");
            for (int j = 0; j < permSize - 1; ++j)
                Console.Write($"{permBlock[i * permSize + j],2}, ");
            Console.WriteLine($"{permBlock[i * permSize + permSize - 1],2}");
        }
    }
}





