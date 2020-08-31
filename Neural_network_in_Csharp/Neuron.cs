using System;
using System.Collections.Generic;
using System.Text;

namespace Neural_network_in_Csharp
{
    public interface Neuron
    {
        public void initialize(float[] weights, float bias);
        public float[] Weights { get; set; }
        public float Bias { get; set; }
    }
}
