using System;
using System.Collections.Generic;
using System.Text;

namespace Neural_network_in_Csharp
{
    interface Loss
    {
        public float CalculateLoss(object predicted, object y, int class_size);

        public object PreprocessY(object y, int class_size);
    }
}
