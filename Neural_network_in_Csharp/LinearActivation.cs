using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Neural_network_in_Csharp
{
    class LinearActivation : Activation
    {

        public float[] CalculateDerivate(object y_pred, object y_real_one_hot)
        {
            float[] predicted_y = (float[])y_pred;
            float[] real_y = (float[])y_real_one_hot;

            List<float> directions = new List<float>();
            for(int i =0; i < real_y.Length; i++)
            {
                directions.Add(real_y[i] - predicted_y[i]);
            }
            return directions.ToArray();
        }

        public List<float> PerformActivation(Neuron[] neurons, float[] previous_neuron_activatios)
        {
            List<float> this_layer_activations = new List<float>();

            foreach (Neuron neuron in neurons)
            {
                float bias = neuron.Bias;
                float[] weights = neuron.Weights;

                float activation = bias;

                // hardcoded linear activation, maybe relu
                for (int weight_i = 0; weight_i < weights.Length; weight_i++)
                {
                    activation += weights[weight_i] * previous_neuron_activatios[weight_i];
                }

                this_layer_activations.Add(activation);
            }

            return this_layer_activations;
        }



    }
}
