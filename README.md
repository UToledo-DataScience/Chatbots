# Chatbots
Dataset: https://raw.githubusercontent.com/google-research-datasets/Taskmaster/master/TM-2-2020/data/movies.json

In this project, we will be implementing a customer assistant chatbot using different types of neural network.

GitHub dataset source: https://github.com/google-research-datasets/Taskmaster

# Contributing

For any changes, make a new branch and pull request in order to merge to `master`.

### Formatting

Install pylint and yapf. e.g. if you're using Ubuntu, `python -m pip install pylint yapf`

Run your files through pylint: `pylint <your file>`

Format your files with yapf: `yapf -i <your file>`

Note that we use [Google's python style standards](https://github.com/google/styleguide/blob/gh-pages/pyguide.md). In particular, follow these naming conventions: `module_name`, `package_name`, `ClassName`, `method_name`, `ExceptionName`, `function_name`, `GLOBAL_CONSTANT_NAME`, `global_var_name`, `instance_var_name`, `function_parameter_name`, `local_var_name`.

### Directory Structure
There should be no python files in the root directory. 

Model classes should be consistent in that they contain the following functions:

- `tf.function(train_step(inputs, target))`: train step wrapped in tf.function for better speed
- `train(dataset, epochs, weight_path=None, word_index=None)`: train loop. `weight_path` denotes path to where weights will be saved, `word_index` determines whether sample will be called at the end of each epoch
- `sample(dataset, word_index)`: return sampled logits using words from the `word_index`

Model files should be structured as follows:
```
...imports...

#subclassed layers to be components in the model
class NewLayer(layers.Layer):
    ...

class NewArchitecture(keras.Model):
    def __init__(self, ...):
        ...
        # NOTE: this should only be defined if your model is sequential in nature
        #       as an example:
        self.net = keras.Sequential([
             layers.LSTM(rnn_units),
             layers.Dense(vocab_size)
        ])
        self.net.summary() # only if the above is defined
        
        #       otherwise the layers of your model should be listed as attributes
        #       of the class
        self.new_layer1 = NewLayer(...)
        
    # NOTE: this should only be defined if your model isn't using keras.Sequential
    def build_graph(self, input_shape):
        inp = keras.Input(shape=input_shape[1:])
        self.call(inp)

        self.build(input_shape)
        self.summary()
```

See the existing architectures for examples.
