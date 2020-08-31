using System;
using System.Collections.Generic;
using System.Text;

namespace Neural_network_in_Csharp
{
    interface Layer<neurontype>
    {
        public void call(Layer<Neuron> layer);

        public float[][] updated_weights { get; set; }

        public void create(int count_neurons, Activation activation, Initializer initializer = null);

        public void pass_forward(float[] values);

        public void pass_backwards(object expected, float lr);

        public void update_weights();

        public Activation activation { get; set; }
        public Initializer initializer { get; set; }

        public void initialize(int model_input=0);

        public bool isInitialized { get; set; }

        public float[] current_activations { get; set; }

        public Layer<Neuron> next_layer { get; set; }

        public Layer<Neuron> caller_layer { get; set; }

        public neurontype[] neurons { get; set; }
    }
}
