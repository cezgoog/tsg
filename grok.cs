using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

namespace Grok
{
    class Grok
    {
        public const string Model = "grok-3-fast-beta"; // "grok-3"
        public const double CostPerMillionPromptTokens = 5.0;
        public const double CostPerMillionCompletionTokens = 25.0;

        private static HttpClient? _httpClient = null;

        private static string AppKey = "";
        private const string ApiUrl = "https://api.x.ai/v1/chat/completions";

        static async Task Main(string[] args)
        {
            AppKey = File.ReadAllText("appkey.txt").Trim();
            string instructionTemplate = File.ReadAllText("instructions.txt");
            string[] symbols = File.ReadLines("symbols.txt").Where(l => !String.IsNullOrWhiteSpace(l)).Select(s => s.Trim().ToUpperInvariant()).Distinct().ToArray();
            Console.WriteLine($"Loaded symbols: {string.Join(", ", symbols)}");

            string objective = File.ReadLines("objective.txt").Where(l => !String.IsNullOrWhiteSpace(l) && Char.IsLetterOrDigit(l[0])).FirstOrDefault() ?? "";
            if (String.IsNullOrWhiteSpace(objective))
            {
                Console.WriteLine("Missing objective. Check if there is a line that starts with a alphanumeric character in objective.txt file");
                return;
            }

            double cost = 0.0;
            Dictionary<string, string> matches = [];
            double[] wcounts = [.. symbols.Select(_ => 0.0)];

            Console.WriteLine($"Objective: {objective}");
            Console.WriteLine();
            Console.WriteLine("SYMB1 SYMB2  WINNER  WCOUNT  COST($)");

            for (int i = 0; i < symbols.Length; i++)
            {
                for (int j = i + 1; j < symbols.Length; j++)
                {
                    string symbol1 = symbols[i];
                    string symbol2 = symbols[j];

                    string instruction = instructionTemplate.Replace("OBJECTIVE", objective);
                    instruction = instruction.Replace("SYMBOL1", symbol1);
                    instruction = instruction.Replace("SYMBOL2", symbol2);

                    (string winner, int promptTokens, int completionTokens) = await GetWinnder(symbol1, symbol2, instruction);
                    cost += promptTokens * CostPerMillionPromptTokens / 1000000.0;
                    cost += completionTokens * CostPerMillionCompletionTokens / 1000000.0;

                    if (winner == symbol1)
                    {
                        wcounts[i] += 1.0;
                        Console.WriteLine($"{symbol1,5} {symbol2,5}  {winner,6}  {wcounts[i],6:0.0}  {cost,7:0.00}");
                        matches[$"{symbol1}:{symbol2}"] = winner;
                    }
                    else if (winner == symbol2)
                    {
                        wcounts[j] += 1.0;
                        Console.WriteLine($"{symbol1,5} {symbol2,5}  {winner,6}  {wcounts[j],6:0.0}  {cost,7:0.00}");
                    }
                    else
                    {
                        winner = ""; // just in case LLM returned something else than a symbol
                        wcounts[i] += 0.5;
                        wcounts[j] += 0.5;
                        Console.WriteLine($"{symbol1,5} {symbol2,5}  {"",6}  {0,6:0.0}  {cost,7:0.00}");
                    }

                    matches[$"{symbol1}:{symbol2}"] = winner;
                }
            }

            Console.WriteLine();
            Console.WriteLine("Scores:");
            IEnumerable<(string symbol, double score)> scores = CalcScores(symbols, wcounts, matches);
            foreach (var x in scores.OrderByDescending(s => s.score))
            {
                Console.WriteLine($"{x.symbol,7} {x.score,7:0.00}");
            }

            Console.WriteLine();
            Console.WriteLine("Win counts:");
            foreach (var x in symbols.Select((s, i) => (symbol: s, count: wcounts[i])).OrderByDescending(s => s.count).ThenBy(s => s.symbol))
            {
                Console.WriteLine($"{x.symbol,7} {x.count,7:0.00}");
            }
        }

        private static IEnumerable<(string symbol, double score)> CalcScores(string[] symbols, double[] wcounts, Dictionary<string, string> matches)
        {
            double avg = wcounts.Average() + 1.0;
            double[] scores = [.. wcounts.Select(c => 1000.0 * (c + 1.0) / avg)];

            for (int iter = 0; iter < 20; iter++)
            {
                double[] newScores = [.. symbols.Select(_ => 1.0)];

                for (int i = 0; i < symbols.Length; i++)
                {
                    string symbol1 = symbols[i];

                    for (int j = i + 1; j < symbols.Length; j++)
                    {
                        string symbol2 = symbols[j];
                        string winner = matches[$"{symbol1}:{symbol2}"];

                        if (winner == "")
                        {
                            // a draw
                            newScores[i] += scores[j] / 3.0;
                            newScores[j] += scores[i] / 3.0;
                        }
                        else if (winner == symbol1)
                        {
                            newScores[i] += scores[j];
                        }
                        else if (winner == symbol2)
                        {
                            newScores[j] += scores[i];
                        }
                    }
                }

                Array.Copy(newScores, 0, scores, 0, scores.Length);
                avg = scores.Average();
                for (int k = 0; k < scores.Length; k++)
                    scores[k] = 1000.0 * scores[k] / avg; // assume the averages score is 1000
            }

            return symbols.Select((s, i) => (symbol: s, score: scores[i]));
        }

        static async Task<(string winner, int promptTokens, int completionTokens)> GetWinnder(string symbol1, string symbol2, string instructions)
        {
            // TODO: there are some weird issues if we keep the httpClient going for a long time. By now dispose the old instance.
            if (_httpClient != null)
                _httpClient.Dispose();
            _httpClient = new();

            // Configure HttpClient
#pragma warning disable CS8602 // Dereference of a possibly null reference.
            _httpClient.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", AppKey);
            _httpClient.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
#pragma warning restore CS8602 // Dereference of a possibly null reference.

            // Create the request payload
            var request = new GrokRequest
            {
                model = Model,
                messages = [new Message { role = "user", content = instructions }]
            };

            bool retry = true;
            int waittime = 1000;

            while (retry)
            {
                retry = false;

                try
                {
                    // Send the request and get the response
                    string responseContent = await SendGrokRequestAsync(request);
                    GrokResponse? response = JsonSerializer.Deserialize<GrokResponse>(responseContent);

                    if (response != null)
                    {
                        string winner = "";

                        if (response?.choices?.Length > 0)
                            winner = response.choices[0].message.content;

                        int promptTokens = 0;
                        int completionTokens = 0;

                        if (response?.usage != null)
                        {
                            promptTokens = response.usage.prompt_tokens;
                            completionTokens = response.usage.completion_tokens;
                        }

                        if (winner.Contains(symbol1) && winner.Contains(symbol2))
                            winner = "";
                        if (winner.Contains(symbol1))
                            winner = symbol1;
                        else if (winner.Contains(symbol2))
                            winner = symbol2;

                        return (winner, promptTokens, completionTokens);
                    }
                    else
                    {
                        Console.WriteLine("Failed to parse the response.");
                    }
                }
                catch (HttpRequestException ex)
                {
                    Console.WriteLine($"HTTP Error: {ex.Message}. Retrying...");
                    Thread.Sleep(waittime);
                    waittime += (waittime >> 1);
                    retry = true;
                }
                catch (JsonException ex)
                {
                    Console.WriteLine($"JSON Parsing Error: {ex.Message}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Unexpected Error: {ex.Message}");
                }
            }

            return ("", 0, 0);
        }

        static async Task<string> SendGrokRequestAsync(GrokRequest request)
        {
            // Serialize the request to JSON
            string jsonRequest = JsonSerializer.Serialize(request);
            var content = new StringContent(jsonRequest, Encoding.UTF8, "application/json");

            // Send the POST request
#pragma warning disable CS8602 // Dereference of a possibly null reference.
            HttpResponseMessage response = await _httpClient.PostAsync(ApiUrl, content);
#pragma warning restore CS8602 // Dereference of a possibly null reference.
            response.EnsureSuccessStatusCode();

            // Read and return the response content
            return await response.Content.ReadAsStringAsync();
        }
    }

    // Define the request model
    public class GrokRequest
    {
        public string model { get; set; } = "";
        public Message[] messages { get; set; } = [];
        public double temperature { get; set; } = 0.0;
    }

    // Define the response model
    public class GrokResponse
    {
        public string id { get; set; } = "";
        public string model { get; set; } = "";
        public Choice[] choices { get; set; } = [];
        public Usage usage { get; set; } = new();
    }

    public class Choice
    {
        public int index { get; set; }
        public Message message { get; set; } = new();
        public string finishReason { get; set; } = "";
    }

    public class Message
    {
        public string role { get; set; } = "";
        public string content { get; set; } = "";
    }

    public class Usage
    {
        public int prompt_tokens { get; set; }
        public int completion_tokens { get; set; }
        public int total_tokens { get; set; }
    }
}
