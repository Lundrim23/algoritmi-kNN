using System;
using System.Collections.Generic;
using System.Linq;

public class KNN
{
    private int _k; // Numri i fqinjeve me te afert
    private List<Tuple<double[], int>> _trainingData; // Lista e te dhenave trajnuese

    public KNN(int k)
    {
        _k = k;
        _trainingData = new List<Tuple<double[], int>>();
    }

    // Metoda per trajnimin e klasifikuesit
    public void Fit(double[][] features, int[] labels)
    {
        for (int i = 0; i < features.Length; i++)
        {
            _trainingData.Add(new Tuple<double[], int>(features[i], labels[i]));
        }
    }

    // Metoda per parashikimin e etiketes se nje pike te re
    public int Predict(double[] newPoint)
    {
        var distances = _trainingData.Select(t => new
        {
            Label = t.Item2,
            Distance = EuclideanDistance(newPoint, t.Item1)
        }).OrderBy(x => x.Distance).Take(_k).ToList();

        return distances.GroupBy(x => x.Label).OrderByDescending(g => g.Count()).First().Key;
    }

    // Metoda per llogaritjen e distances Euclidean
    private double EuclideanDistance(double[] point1, double[] point2)
    {
        double sum = 0;
        for (int i = 0; i < point1.Length; i++)
        {
            sum += Math.Pow(point1[i] - point2[i], 2);
        }
        return Math.Sqrt(sum);
    }
}

class Program
{
    static void Main()
    {
        // Te dhenat trajnuese
        double[][] features = new double[][]
        {
            new double[] {1, 2},
            new double[] {3, 4},
            new double[] {5, 6},
            new double[] {7, 8}
        };
        int[] labels = new int[] {0, 1, 0, 1};

        // Krijimi dhe trajnimi i klasifikuesit kNN
        KNN knn = new KNN(k: 3);
        knn.Fit(features, labels);

        // Pika e re per parashikim
        double[] newPoint = new double[] {4, 5};

        // Parashikimi i etiketes per piken e re
        int predictedLabel = knn.Predict(newPoint);

        // Shfaqja e rezultatit
        Console.WriteLine($"Etiketa e parashikuar për piken ({newPoint[0]}, {newPoint[1]}): {predictedLabel}");
    }
}
