import re
from tensorflow.keras.applications import Xception

def pre_trained_weights_transfer(model, depth=8):
    depth_block_map = ['block5', 'block6', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12']
    patterns = depth_block_map[depth:]

    Xception_model = Xception(weights='imagenet', include_top=False)
    custom_weights = model.get_weights()
    
    extracted_weights = []
    for layer in Xception_model.layers:
        if any(re.search(pattern, layer.name) for pattern in patterns):
            continue
        extracted_weights += layer.get_weights()

    custom_weights[:-2] = extracted_weights
    model.set_weights(custom_weights)

    return model
