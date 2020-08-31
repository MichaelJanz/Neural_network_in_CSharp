using System;
using System.Collections.Generic;
using System.Text;

namespace Neural_network_in_Csharp
{
    public interface Initializer
    {
        public void initialize_weights(Neuron[] neurons, int neurons_previous_layer, float mean = 0, float stdDev = 1);

        public void initialize_bias(Neuron[] neurons, float mean = 0, float stdDev = 1);

    }
}
