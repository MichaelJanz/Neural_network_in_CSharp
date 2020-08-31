using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Text;

namespace Neural_network_in_Csharp
{
    class Model
    {
        List<Layer<Neuron>> layer_list = new List<Layer<Neuron>>();

        public void Add(Layer<Neuron> layer)
        {
            layer_list.Add(layer);
        }

        public void Create(Loss loss_function, int input_size, float learning_rate = 0.01f)
        {
            this.loss_function = loss_function;
            this.learning_rate = learning_rate;

            foreach(var layer in layer_list)
            {
                if (!layer.isInitialized)
                {
                    layer.initialize(input_size);
                }
            }
        }

        float learning_rate { get; set; }

        Loss loss_function { get; set; }
        public void Train(float[][] x, int[] y, int epochs=1)
        {
            Layer<Neuron> first_layer = layer_list[0];
            Layer<Neuron> last_layer = layer_list[layer_list.Count - 1];

            // check if the first layer has as many values as x:

            if(first_layer.neurons.Length != x[0].Length)
            {
                throw new Exception("Dimension of layer 0 (" + first_layer.neurons.Length + ") must match to x ("+ x.Length + ")");
            }

            // For each epoch:
            float loss = 0;
            for (int epoch=0; epoch < epochs; epoch++)
            {

                for (int i =0;i < y.Length;i++)
                {
                    float[] x_train = x[i];
                    first_layer.pass_forward(x_train);
                    float[] predicted = last_layer.current_activations;

                    object y_processed = loss_function.PreprocessY(y[i], last_layer.neurons.Length);
                    //calculate the loss after each finished step
                    loss = loss_function.CalculateLoss(predicted, y_processed, last_layer.neurons.Length);

                    // Start backpropagation by calculating the directions, where the neurons have to be pushed




                    // Start backpropagation in the last layer

                    last_layer.pass_backwards(y_processed, learning_rate);
                    foreach(var layer in layer_list)
                    {
                        layer.update_weights();
                    }
                    //Backpropagation is finished
                }

                Console.WriteLine("Loss:" + Convert.ToString(loss));


            }

            Console.WriteLine("Training is finished");
            Console.WriteLine("Final loss is:" + loss);

            Console.ReadLine();
        }
    }
}
