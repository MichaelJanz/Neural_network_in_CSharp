using System;
using System.Collections.Generic;
using System.Text;

namespace Neural_network_in_Csharp
{
    public class Simple_neuron : Neuron
    {
        public void initialize(float[] weights, float bias)
        {
            this.Weights = weights;
            this.Bias = bias;
        }

        public float[] Weights { get; set; }
        public float Bias { get; set; }

    }
}
