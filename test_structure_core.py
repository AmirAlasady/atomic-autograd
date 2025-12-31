import sys
import os
import numpy as np

# Ensure the current directory is in python path
sys.path.append(os.getcwd())

print("Testing atomic imports...")

try:
    import atomic
    print(f"Successfully imported atomic: {atomic}")
    
    # Test top-level exports
    from atomic import Tensor, Base_Layer
    print("Successfully imported Tensor and Base_Layer from top-level")
    
    # Test core paths
    from atomic.core.autograd import Tensor as CoreTensor
    from atomic.core.basestructure import Base_Layer as CoreBase
    print("Successfully imported Tensor and Base_Layer from core")
    
    # Test submodules which should now point to core
    from atomic.layers import Dense, Conv2D
    print("Successfully imported layers")
    
    from atomic.models import Sequential, BaseModel
    print("Successfully imported models")
    
    from atomic.losses import MSELoss, CrossEntropy
    print("Successfully imported losses")
    
    from atomic.optimizers import Adam, SGD
    print("Successfully imported optimizers")
    
    from atomic.activations import ReLU, Sigmoid
    print("Successfully imported activations")
    
    from atomic.nlp import Transformer
    print("Successfully imported Transformer")
    
    from atomic.metrics import Accuracy
    print("Successfully imported Accuracy")

    # Test detailed functionality
    print("\nTesting Tensor functionality...")
    t = Tensor([1, 2, 3], requires_grad=True)
    t2 = t * 2
    t2.backward(Tensor([1, 1, 1]))
    print(f"Tensor grad: {t.grad.data}")
    assert np.allclose(t.grad.data, [2, 2, 2])
    
    print("\nTesting Dense layer...")
    d = Dense(3, 2)
    out = d(Tensor(np.random.randn(5, 3)))
    print(f"Dense output shape: {out.shape}")
    assert out.shape == (5, 2)

    print("\nAll tests passed!")
except Exception as e:
    print(f"\nFAILED: {e}")
    import traceback
    traceback.print_exc()
