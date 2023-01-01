using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;
using System.Diagnostics;
using System.Threading.Tasks;
using Accord.Imaging;
using Accord.Math;
using Accord.MachineLearning;
using Accord.Statistics.Filters;
using System.Data;

namespace PH
{
    public class DescriptorBase
    {        
        FCTH_Descriptor.FCTH fcth;
        CEDD_Descriptor.CEDD cedd;
        SpeededUpRobustFeaturesDetector surf;
        KMeans kmeans;

        public DescriptorBase()
        {
            fcth = new FCTH_Descriptor.FCTH();            
            cedd = new CEDD_Descriptor.CEDD();
            surf = new SpeededUpRobustFeaturesDetector();
            kmeans = new KMeans(k: 400);
        }

        public List<double[]> ComputeFCTHandSave(List<string> imagePaths, string fileType)
        {
            List<double[]> fcthList = new List<double[]>();
            Console.WriteLine("FCTH features are being extracted for {0}...", fileType);
            string fileName = @"pre-computed\precomputed_FCTH_" + fileType + ".csv";
            Helpers.WriteHeaderToCSV(fileName, 144);

            Stopwatch stopWatch = new Stopwatch();
            stopWatch.Start();

            foreach (string imgPath in imagePaths)
            {
                Bitmap bitmapImg = new Bitmap(imgPath);
                double[] fcthTable = fcth.Apply(bitmapImg, 2);
                string imgLabel = Helpers.GetLabel(imgPath);
                Helpers.SaveArrayAsCSV(fcthTable, fileName, imgLabel);
                fcthList.Add(fcthTable);
            }

            stopWatch.Stop();
            TimeSpan ts = stopWatch.Elapsed;
            double elapsedTime = Math.Round(ts.TotalSeconds, 2);

            Console.WriteLine("Done. precomputed_FCTH_{0}.csv is regenerated in {1} seconds", fileType, elapsedTime);
            return fcthList;
        }

        public List<double[]> ComputeCEDDandSave(List<string> imagePaths, string fileType)
        {
            List<double[]> ceddList = new List<double[]>();
            Console.WriteLine("CEDD features are being extracted for {0}...", fileType);
            string fileName = @"pre-computed\precomputed_CEDD_" + fileType + ".csv";
            Helpers.WriteHeaderToCSV(fileName, 192);

            Stopwatch stopWatch = new Stopwatch();
            stopWatch.Start();

            foreach (string imgPath in imagePaths)
            {
                Bitmap bitmapImg = new Bitmap(imgPath);
                double[] ceddTable = cedd.Apply(bitmapImg);
                string imgLabel = Helpers.GetLabel(imgPath);
                Helpers.SaveArrayAsCSV(ceddTable, fileName, imgLabel);
                ceddList.Add(ceddTable);
            }

            stopWatch.Stop();
            TimeSpan ts = stopWatch.Elapsed;
            double elapsedTime = Math.Round(ts.TotalSeconds, 2);

            Console.WriteLine("Done. precomputed_CEDD_{0}.csv is regenerated in {1} seconds", fileType, elapsedTime);
            return ceddList;
        }

        public Tuple<double[,], KMeansClusterCollection> ComputeSURFForTrainSetandSave(List<string> trainImagePaths)
        {
            List<double[]> vStackedTrainDescriptors = new List<double[]>();
            List<double[][]> trainDescriptors = new List<double[][]>();
            List<string> trainLabels = new List<string>();
            int imgCount = 0;

            string fileName = @"pre-computed\precomputed_SURF_train.csv";
            Helpers.WriteHeaderToCSV(fileName, 400);
            Console.WriteLine("SURF features are being extracted for train...");

            Stopwatch stopWatch = new Stopwatch();
            stopWatch.Start();

            foreach (string imgPath in trainImagePaths)
            {
                imgCount++;
                string imgLabel = Helpers.GetLabel(imgPath);
                trainLabels.Add(imgLabel);
                Bitmap bitmapImg = new Bitmap(imgPath);
                List<SpeededUpRobustFeaturePoint> descriptors = surf.ProcessImage(bitmapImg);
                double[][] surfTable = descriptors.Apply(d => d.Descriptor);
                trainDescriptors.Add(surfTable);

                foreach (double[] item in surfTable) //vstack
                {
                    vStackedTrainDescriptors.Add(item);
                }
            }
            KMeansClusterCollection kmeans = ClusterDescriptors(vStackedTrainDescriptors.ToArray());
            double[,] allTrainFeaturesBoVW = ExtractFeatures(kmeans, trainDescriptors, imgCount);
            int labelCtr = 0;
            for (int i = 0; i < allTrainFeaturesBoVW.GetLength(0); i++)
            {
                Helpers.SaveArrayAsCSV(allTrainFeaturesBoVW.GetRow(i), fileName, trainLabels[labelCtr]);
                labelCtr++;
            }

            stopWatch.Stop();
            TimeSpan ts = stopWatch.Elapsed;
            double elapsedTime = Math.Round(ts.TotalSeconds, 2);
            Console.WriteLine("Done. precomputed_SURF_train.csv is regenerated in {0} seconds", elapsedTime);

            return Tuple.Create(allTrainFeaturesBoVW,kmeans);
        }

        public double[,] ComputeSURFForTestSetandSave(List<string> valImagePaths, KMeansClusterCollection kmeans)
        {
            List<double[]> vStackedTestDescriptors = new List<double[]>();
            List<double[][]> valDescriptors = new List<double[][]>();
            List<string> valLabels = new List<string>();
            int imgCount = 0;

            string fileName = @"pre-computed\precomputed_SURF_val.csv";
            Helpers.WriteHeaderToCSV(fileName, 400);
            Console.WriteLine("SURF features are being extracted for val...");

            Stopwatch stopWatch = new Stopwatch();
            stopWatch.Start();

            foreach (string imgPath in valImagePaths)
            {
                imgCount++;
                string imgLabel = Helpers.GetLabel(imgPath);
                valLabels.Add(imgLabel);
                Bitmap bitmapImg = new Bitmap(imgPath);
                List<SpeededUpRobustFeaturePoint> descriptors = surf.ProcessImage(bitmapImg);
                double[][] surfTable = descriptors.Apply(d => d.Descriptor);
                valDescriptors.Add(surfTable);

                foreach (double[] item in surfTable) //vstack
                {
                    vStackedTestDescriptors.Add(item);
                }
            }
            double[,] allValFeaturesBoVW = ExtractFeatures(kmeans, valDescriptors, imgCount);
            int labelCtr = 0;
            for (int i = 0; i < allValFeaturesBoVW.GetLength(0); i++)
            {
                Helpers.SaveArrayAsCSV(allValFeaturesBoVW.GetRow(i), fileName, valLabels[labelCtr]);
                labelCtr++;
            }

            stopWatch.Stop();
            TimeSpan ts = stopWatch.Elapsed;
            double elapsedTime = Math.Round(ts.TotalSeconds, 2);
            Console.WriteLine("Done. precomputed_SURF_val.csv is regenerated in {0} seconds", elapsedTime);
            return allValFeaturesBoVW;
        }

        public KMeansClusterCollection ClusterDescriptors(double[][] input)
        {
            // Compute and retrieve the data centroids
            // quantization
            KMeansClusterCollection clusters = kmeans.Learn(input);
            // Use the centroids to parition all the data
            // // // int[] labels = clusters.Decide(input);
            return clusters;
        }

        public double[,] ExtractFeatures(KMeansClusterCollection kmeans, List<double[][]> descriptors, int imageCount)
        {
            int[,] totalFeatures = new int[imageCount,400];

            for (int i = 0; i < imageCount; i++)
            {
                for (int j = 0; j < descriptors[i].Length; j++)
                {
                    var feature = descriptors[i][j];
                    int idx = kmeans.Decide(feature);
                    //POOLING
                    totalFeatures[i, idx] += 1;
                }
            }

            return NormalizeBoWV(totalFeatures);
        }

        public static double[,] NormalizeBoWV(int[,] totalFeatures)
        {
            // Create the aforementioned sample table
            DataTable table = new DataTable("Data to Normalize");

            for (int i = 1; i < 401; i++)
            {
                string columnName = "f" + i.ToString();
                table.Columns.Add(columnName, typeof(double));
            }

            for (int i = 0; i < totalFeatures.GetLength(0); i++)
            {
                DataRow row = table.NewRow();

                for (int j = 0; j < totalFeatures.GetRow(i).Length; j++)
                {
                    row["f" + (j + 1).ToString()] = totalFeatures[i, j];
                }
                table.Rows.Add(row);
            }

            // The filter will ignore non-real (continuous) data
            Normalization normalization = new Normalization(table);

            // Now we can process another table at once:
            DataTable result = normalization.Apply(table);

            // The result will be a table with the same columns, but
            // in which any column named "Age" will have been normalized
            // using the previously detected mean and standard deviation:
            double[,] normalizedArray = new double[result.Rows.Count, result.Columns.Count];

            for (int rowIndex = 0; rowIndex < result.Rows.Count; rowIndex++)
            {
                for (int columnIndex = 0; columnIndex < result.Columns.Count; columnIndex++)
                {
                    normalizedArray[rowIndex, columnIndex] = Convert.ToDouble(result.Rows[rowIndex][columnIndex]);
                }
            }
            return normalizedArray;
        }
    }
}
