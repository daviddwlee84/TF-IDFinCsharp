using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TFIDFExample
{
    class Program
    {
        static void Main(string[] args)
        {
            // Some example documents.
            string[] documents =
            {
                "The sun in the sky is bright.",
                "We can see the shining sun, the bright sun.",
                "Seven States Hit Hardest by Gas Prices All Have This in Common",
                "Here's how oil and gas prices could be affected by Russia's invasion of Ukraine (CNBC)"
            };

            // Apply TF*IDF to the documents and get the resulting vectors.
            TFIDF.InitIDFTable(documents, 0);
            double[][] inputs = TFIDF.Transform(documents);
            inputs = TFIDF.Normalize(inputs);

            // Display the _vocabularyIDF
            // (not sure why we can't create the TFIDF instance after change it to non-static)
            Console.WriteLine($"{TFIDF._vocabularyIDF}\n");
            foreach (KeyValuePair<string, double> kvp in TFIDF._vocabularyIDF)
            {
                Console.WriteLine($"{kvp.Key}: {kvp.Value}");
            }

            // Display the output.
            for (int index = 0; index < inputs.Length; index++)
            {
                Console.WriteLine(documents[index]);

                foreach (double value in inputs[index])
                {
                    Console.Write(value + ", ");
                }

                Console.WriteLine("\n");
            }

            // Console.WriteLine($"{TFIDF.GetLang("enus")}");
            Console.WriteLine($"{TFIDF.GetLang("en-us")}");

            // Console.WriteLine("Press any key ..");
            // Console.ReadKey();
        }
    }
}
