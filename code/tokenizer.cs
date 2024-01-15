// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// Open Package-Manager-Console and run this first:
// NuGet\Install-Package Microsoft.DeepDev.TokenizerLib -Version 1.3.3
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

// https://github.com/stephentoub/Tokenizer
// https://www.nuget.org/packages/Microsoft.DeepDev.TokenizerLib/

using Microsoft.DeepDev;

var IM_START = "<|im_start|>";
var IM_END = "<|im_end|>";
var specialTokens = new Dictionary<string, int> { { IM_START, 100264 }, { IM_END, 100265 }, };
ITokenizer tokenizer = await TokenizerBuilder.CreateByModelNameAsync("gpt-4", specialTokens);

string text = "Hello World";
// demo 1 encode tokens
StringToTokens(IM_START + text + IM_END);
// demo 2 decode tokens
PrintDecodedTokens(0, 5);
PrintDecodedTokens(9905, 5);
PrintDecodedTokens(100261, 5);
// demo 3 save all tokens into one file
CreateOutputTokensFile(@"C:\tokenizer\output.txt", 100266);

void PrintDecodedTokens(int start, int take)
{
    Console.WriteLine($"\nPrint {take} tokens from {start} to {start + take - 1}:");
    for (int i = start; i < start + take; i++)
    {
        string decoded = tokenizer.Decode(new[] { i });
        Console.WriteLine($"{i} = {decoded}");
    }
}
void StringToTokens(string text)
{
    Console.WriteLine("\nString to encode:\n" + text + "\n");
    var tokens = tokenizer.Encode(text, new HashSet<string>(specialTokens.Keys));

    Console.WriteLine($"Tokens encoded = {tokens.Count}:");
    foreach (var token in tokens)
    {
        string tokenName = tokenizer.Decode(new[] { token });
        Console.WriteLine($"{token} ({tokenName})");
    }

    var decoded = tokenizer.Decode(tokens.ToArray());
    Console.WriteLine("\nDecoded tokens\n" + decoded);
}
void CreateOutputTokensFile(string outputPath, int length)
{
    Console.WriteLine("\nSaved " + length + " tokens in " + outputPath);
    using (StreamWriter writer = new StreamWriter(outputPath))
    {
        for (int i = 0; i < length; i++)
        {
            var decoded = tokenizer.Decode(new[] { i });
            writer.WriteLine(decoded);
        }
    }
    Console.WriteLine("Tokenizer file created successfully");
}
