using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using NAudio.Wave;
using Accord.MachineLearning;
using MathNet.Numerics.IntegralTransforms;
using System.Text.Json;
using Accord.Audio;
using Accord.Audio.Windows;
using Accord.Math.Distances;

class Program
{
    private static double[] currentFeatures;
    private static KNearestNeighbors knn;
    private static List<double[]> trainingFeatures = new List<double[]>();
    private static List<int> trainingLabels = new List<int>();

    private const int SAMPLE_RATE = 44100;
    private const int FRAME_SIZE = 512;
    private const int NUM_COEFFICIENTS = 13;

    static async Task Main(string[] args)
    {
        Console.WriteLine("Speaker identification");
        Console.WriteLine("----------------------");

        while (true)
        {
            Console.WriteLine("\n please select an option .");
            Console.WriteLine("1. load an audio file ");
            Console.WriteLine("2. add a training sample");
            Console.WriteLine("3. train model ");
            Console.WriteLine("4. test model ");
            Console.WriteLine("5. exit ");

            string choice = Console.ReadLine();

            switch (choice)
            {
                case "1":
                    await LoadAudioFile();
                    break;
                case "2":
                    AddTrainingSample();
                    break;
                case "3":
                    TrainModel();
                    break;
                case "4":
                    TestModel();
                    break;
                case "5":
                    return;
                default:
                    Console.WriteLine("Invalid option !");
                    break;
            }
        }
    }
    private static async Task LoadAudioFile()
    {
        Console.WriteLine("Please enter the full path to the audio file:");
        string filePath = Console.ReadLine();

        if (!File.Exists(filePath))
        {
            Console.WriteLine("File was not found!");
            return;
        }

        try
        {
            float[] samples = LoadAudio(filePath);
            if (samples != null)
            {
                currentFeatures = ExtractMFCC(samples, SAMPLE_RATE);
                if (currentFeatures != null)
                {
                    Console.WriteLine("Audio file processed successfully.");
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
        }
    }


    private static void AddTrainingSample()
    {
        if (currentFeatures == null)
        {
            Console.WriteLine("Please load an audio file first !");
            return;
        }

        Console.WriteLine("Please enter the speakers numeric label ");
        if (int.TryParse(Console.ReadLine(), out int label))
        {
            trainingFeatures.Add(currentFeatures);
            trainingLabels.Add(label);
            Console.WriteLine($"training model {trainingFeatures.Count} with label {label} added .");
        }
        else
        {
            Console.WriteLine("The entered label is not valid .");
        }
    }

    private static void TrainModel()
    {
        if (trainingFeatures.Count < 2)
        {
            Console.WriteLine("At least two test samples are required .");
            return;
        }

        try
        {
            int k = Math.Min(3, trainingFeatures.Count);
            knn = new KNearestNeighbors(k, trainingFeatures.ToArray(), trainingLabels.ToArray());
            Console.WriteLine("model was successfully trained .");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error in trainig model : {ex.Message}");
        }
    }

    private static void TestModel()
    {
        if (knn == null)
        {
            Console.WriteLine("Please train the model first .");
            return;
        }

        if (currentFeatures == null)
        {
            Console.WriteLine("Please load an audio file for testing first .");
            return;
        }

        try
        {
            int predictedLabel = knn.Decide(currentFeatures);
            Console.WriteLine($"Speaker identified : {predictedLabel}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Speaker identification error : {ex.Message}");
        }
    }
    private static float[] LoadAudio(string wavPath)
    {
        using (var reader = new AudioFileReader(wavPath))
        {
            const int sampleRate = 44100; 
            if (reader.WaveFormat.SampleRate != sampleRate)
                throw new Exception("The sample rate of the file does not match SAMPLE_RATE!");

            
            int sampleCount = (int)(reader.Length / (reader.WaveFormat.BitsPerSample / 8));
            float[] samples = new float[sampleCount];

            
            int samplesRead = reader.Read(samples, 0, sampleCount);
            if (samplesRead != sampleCount)
                throw new Exception("Error reading all samples!");

            return samples;
        }
    }
    private static void SaveTrainingData(string filePath)
    {
        var data = new { Features = trainingFeatures, Labels = trainingLabels };
        File.WriteAllText(filePath, JsonSerializer.Serialize(data));
    }
    public static double[] ExtractMFCC(float[] samples, int sampleRate)
    {
        const int FRAME_SIZE = 512;         
        const int NUM_COEFFICIENTS = 13;    

        var mfcc = new MelFrequencyCepstrumCoefficient(FRAME_SIZE, NUM_COEFFICIENTS);

        double[] doubleSample = Array.ConvertAll(samples , x =>(double)x);

        Signal signal = Signal.FromArray(doubleSample, sampleRate);

        var frames = mfcc.Transform(signal);
        //--------------------------------------------------
        double[] meanMFCC = new double[NUM_COEFFICIENTS];
        for (int i = 0; i < NUM_COEFFICIENTS; i++)
        {
            meanMFCC[i] = frames.Average(frame => frame[i]);
        }

        return meanMFCC;
    }
}
