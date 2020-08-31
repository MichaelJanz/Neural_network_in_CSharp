using System;
using System.Runtime.Serialization;

namespace Neural_network_in_Csharp
{
    class Program
    {
        static void Main(string[] args)
        {


            // Hardcoded normalization for x to prevent exploding/vanishing gradients
            float max = 7.8f;

            float[][] x = new float[][]
            {
                new float[]
                {
                    0/max,
                    0.2f/max,
                    0.4f/max,
                    0.7f/max
                },
                new float[]
                {
                    1/max,
                    1.7f/max,
                    1.3f/max,
                    1.1f/max
                },
                new float[]
                {
                    2.3f/max,
                    7.8f/max,
                    5.2f/max,
                    3.1f/max
                }};



            int[] y = new int[] { 0,1,2};

            DenseLayer input_layer = new DenseLayer(4, new LinearActivation());
            DenseLayer hidden1 = new DenseLayer(1);
            DenseLayer output_layer = new DenseLayer(3, new SoftmaxActivation());

            input_layer.call(hidden1);
            hidden1.call(output_layer);

            Model model = new Model();

            model.Add(input_layer);
            model.Add(hidden1);
            model.Add(output_layer);

            model.Create(new loss_sparse_categorical_crossentropy(), 4);
            model.Train(x, y, 100);
        }
    }
}
