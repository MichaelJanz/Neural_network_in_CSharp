using System;
using System.Collections.Generic;
using System.Text;

namespace Neural_network_in_Csharp
{
    interface Activation
    {
        public List<float> PerformActivation(Neuron[] neurons, float[] previous_neuron_activations);

        public float[] CalculateDerivate(object y_predicted, object y_real_one_hot);

    }
}
