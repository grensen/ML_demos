// https://stackoverflow.com/questions/25134897/gzip-compression-and-decompression-in-c-sharp

using System.IO.Compression;
using System.Text;

var input = "This is a test. This is a test. ";
input += input;
input += input;
input += input;

Console.WriteLine("original: " + input.Length);
Console.WriteLine(input + "\n");

byte[] compressed = Compress(input);
Console.WriteLine("compressed: " + compressed.Length);
Console.WriteLine(Convert.ToBase64String(compressed) + "\n");

string decompressed = Decompress(compressed);
Console.WriteLine("decompressed: " + decompressed.Length);
Console.WriteLine(decompressed);

static byte[] Compress(string input)
{
    byte[] encoded = Encoding.UTF8.GetBytes(input);
    using (var result = new MemoryStream())
    {
        using (var compressionStream = new BrotliStream(result, CompressionLevel.Optimal))
            compressionStream.Write(encoded, 0, encoded.Length);
        return result.ToArray();
    }
}
static string Decompress(byte[] input)
{
    using (var source = new MemoryStream(input))
        using (var decompressedResult = new MemoryStream())
        {
            using (var decompressionStream = new BrotliStream(source, CompressionMode.Decompress))
                decompressionStream.CopyTo(decompressedResult);
            byte[] decompressedData = decompressedResult.ToArray();
            return Encoding.UTF8.GetString(decompressedData);
        }
}