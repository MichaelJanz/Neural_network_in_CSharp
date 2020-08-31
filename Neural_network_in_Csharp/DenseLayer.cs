using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text;

namespace Neural_network_in_Csharp
{
    class DenseLayer : Layer<Neuron>
    {
        public DenseLayer(int count_neurons, Activation activation = null, Initializer initializer = null)
        {
            create(count_neurons, activation, initializer); 
        }

        public Neuron[] neurons { get; set; }

        public Layer<Neuron> next_layer { get; set; }
        public Layer<Neuron> caller_layer { get; set; }

        public Activation activation { get; set; }
        public Initializer initializer { get; set; }
        public float[] current_activations { get; set; }
        public float[][] updated_weights { get; set; }
        public bool isInitialized { get; set; }

        public void pass_forward(float[] neuron_activations)
        {

            List<float> this_layer_activations = activation.PerformActivation(this.neurons, neuron_activations);
            this.current_activations = this_layer_activations.ToArray();
            if(next_layer != null)
            {
                next_layer.pass_forward(this_layer_activations.ToArray());
            }
            else { return; }
        }

        public void pass_backwards(object expected, float lr)
        {

            float[] predicted = this.current_activations;

            float[] target_vals = (float[])expected;

            float[] directions = this.activation.CalculateDerivate(predicted, target_vals);

            List<float[]> updated_weights = new List<float[]>();
            for (int i = 0; i < directions.Length; i++)
            {
                float neuron_activation = directions[i];
                Neuron neuron = this.neurons[i];

                float sum_weights = neuron.Weights.Sum();

                // Adjust the weights of the neurons
                float[] temp_weights = new float[neuron.Weights.Length];
                for (int weight_i = 0; weight_i < neuron.Weights.Length; weight_i++)
                {
                    temp_weights[weight_i] = weight_i + ((weight_i / sum_weights) * lr);
                }
                updated_weights.Add((float[])temp_weights.Clone());
            }

            this.updated_weights = updated_weights.ToArray();

            if (caller_layer != null)
            {
                caller_layer.pass_backwards(this.current_activations, lr);
            }
            else
            {
                return;
            }
        }

        public void call(Layer<Neuron> layer_to_call)
        {

            this.next_layer = layer_to_call;
            this.next_layer.caller_layer = this;
        }

        public void create(int count_neurons, Activation activation, Initializer initializer=null)
        {
            if (initializer == null)
            {
               this.initializer = new Initializer_random_norm();
            }
            else
            {
                this.initializer = initializer;
            }

            if(activation == null)
            {
                this.activation = new LinearActivation();
            }
            else
            {
                this.activation = activation;
            }
            this.neurons = new Simple_neuron[count_neurons];
            
            for(int i =0; i < this.neurons.Length; i++)
            {
                this.neurons[i] = new Simple_neuron();
            }
        }

        public void update_weights()
        {
            if(this.updated_weights != null)
            {
                for(int i = 0; i < updated_weights.Length; i++)
                {
                    this.neurons[i].Weights = updated_weights[i];
                }
            }
        }

        public void initialize(int model_input=0)
        {
            int count_weights = 0;
            if(caller_layer == null)
            {
                count_weights = model_input;
                for(int i =0;i < neurons.Length; i++)
                {
                    neurons[i].Weights = new float[] { 1 };
                    neurons[i].Bias = 0f;
                }

            }
            else
            {
                count_weights = caller_layer.neurons.Length;
                this.initializer.initialize_weights(this.neurons, count_weights);
                this.initializer.initialize_bias(this.neurons);
            }

            this.isInitialized = true;
        }
    }
}
