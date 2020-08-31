using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace Neural_network_in_Csharp
{
    class SoftmaxActivation : Activation
    {
        public float[] CalculateDerivate(object y_pred, object y_real_one_hot)
        {
            float[] predicted_y = (float[])y_pred;
            float[] real_y = (float[])y_real_one_hot;

            List<float> directions = new List<float>();
            for (int i = 0; i < real_y.Length; i++)
            {
                directions.Add(real_y[i] - predicted_y[i]);
            }
            return directions.ToArray();
        }

        public List<float> PerformActivation(Neuron[] neurons, float[] previous_neuron_activations)
        {
            List<float> this_layer_activations = new List<float>();

            foreach (Neuron neuron in neurons)
            {
                float bias = neuron.Bias;
                float[] weights = neuron.Weights;

                float activation = bias;

                // activation for softmax
                for (int weight_i = 0; weight_i < weights.Length; weight_i++)
                {
                    activation += weights[weight_i] * previous_neuron_activations[weight_i];
                }
                // e pow activation here
                this_layer_activations.Add(Convert.ToSingle(Math.Pow(Math.E, activation)));
            }

            //applying softmax
            float sum_activations = this_layer_activations.Sum();
            for (int i = 0; i < this_layer_activations.Count; i++)
            {
                this_layer_activations[i] = this_layer_activations[i] / sum_activations;
            }

            return this_layer_activations;
        }
    }
}
