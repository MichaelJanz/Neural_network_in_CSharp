using System;
using System.Collections.Generic;
using System.Text;

namespace Neural_network_in_Csharp
{
    class Initializer_random_norm : Initializer
    {
        Random rand = new Random();

        // Function based on stackoverflow comment by yoyoyoyosef:
        // https://stackoverflow.com/questions/218060/random-gaussian-variables
        private float RandomValueByGauss(float mean = 0, float stdDev = 1)
        {
            double u1 = 1.0 - rand.NextDouble();
            double u2 = 1.0 - rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            double randNormal = mean + stdDev * randStdNormal;

            return (float)randNormal;
        }

        public void initialize_weights(Neuron[] neurons, int neurons_previous_layer, float mean=0, float stdDev=1)
        {

            for(int i =0; i< neurons.Length; i++)
            {
                List<float> weights = new List<float>();

                for(int w_index = 0; w_index < neurons_previous_layer; w_index++)
                {
                    weights.Add(RandomValueByGauss(mean, stdDev));    
                }
                neurons[i].Weights = weights.ToArray();
            }
        }

        public void initialize_bias(Neuron[] neurons, float mean, float stdDev)
        {
            for (int i = 0; i < neurons.Length; i++)
            {

                neurons[i].Bias = RandomValueByGauss(mean, stdDev);
            }
        }

    }
}
