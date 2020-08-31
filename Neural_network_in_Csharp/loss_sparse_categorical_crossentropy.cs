using System;
using System.Collections.Generic;
using System.Text;

namespace Neural_network_in_Csharp
{
    class loss_sparse_categorical_crossentropy : Loss
    {

        public float[] To_one_hot(int y, int class_size)
        {
            float[] one_hot = new float[class_size];

            one_hot[y] = 1f;

            return one_hot;
        }

        public object PreprocessY(object y, int class_size)
        {
            int y_real = (int)y;
            return To_one_hot(y_real, class_size);
        }
        public float CalculateLoss(object predicted, object y_onehot, int class_size)
        {
            float[] predicted_vals = (float[])predicted;
            float[] real_y_oh = (float[])y_onehot;
            //One-hot encode y on the fly

            float loss = 0;
            //Got the loss from here: https://github.com/keras-team/keras/issues/6444
            for (int i = 0; i < class_size; i++)
            {
                loss += Convert.ToSingle(real_y_oh[i] * Math.Log(predicted_vals[i]));
            }

            return loss;

        }
    }
}
