use crate::activations::Activation;
use avance::AvanceBar;
use matrix::matrix::Matrix;

/// The main neural network struct, containing the configuration and state of the network.
///
/// This struct represents a neural network with a configurable number of layers, activation function, and learning rate.
/// It stores the weights and biases for each layer, as well as the input data and activation function to be used.
pub struct Network {
    /// The number of neurons in each layer of the network.
    layers: Vec<usize>,
    /// The weights for each connection between neurons.
    weights: Vec<Matrix>,
    /// The biases for each neuron.
    biases: Vec<Matrix>,
    /// The input data for the network.
    data: Vec<Matrix>,
    /// The activation function to use for the network.
    activation: Activation,
    /// The learning rate to use for the network.
    learning_rate: f64,
}

impl Network {

    /// Creates a new neural network with the specified layer sizes, activation function, and learning rate.
    ///
    /// # Arguments
    /// * `layers` - A vector of usize values representing the number of neurons in each layer of the network.
    /// * `activation` - The activation function to use for the network.
    /// * `desired_learning_rate` - The learning rate to use for the network.
    ///
    /// # Returns
    /// A new `Network` instance with the specified configuration.
    pub fn new(layers: Vec<usize>, activation: Activation, desired_learning_rate: f64) -> Self {
        let mut weights: Vec<Matrix> = vec![];
        let mut biases: Vec<Matrix> = vec![];
        for i in 0..layers.len() - 1 {
            weights.push(Matrix::random(layers[i + 1], layers[i]));
            biases.push(Matrix::random(layers[i + 1], 1));
        }
        let default_learning_rate: f64 = 0.5;
        let learning_rate: f64 = if default_learning_rate != desired_learning_rate {
            desired_learning_rate
        } else {
            default_learning_rate
        };
        Network {
            layers,
            weights,
            biases,
            data: vec![],
            activation,
            learning_rate, /*map_with_learning_rate*/
        }
    }

    /// Performs a forward pass through the neural network.
    ///
    /// # Arguments
    /// * `inputs` - A `Matrix` containing the input data for the network.
    ///
    /// # Returns
    /// A `Matrix` containing the output of the neural network after the forward pass.
    pub fn feed_forward(&mut self, inputs: Matrix) -> Matrix {
        assert!(
            self.layers[0] == inputs.data.len(),
            "Invalid Number of Inputs"
        );
        //   println!("{:?} {:?}",self.weights[0],inputs);
        //   println!("{:?}",self.weights[0].dot_multiply(&inputs).add(&self.biases[0]));
        let mut current = inputs;
        self.data = vec![current.clone()];
        for i in 0..self.layers.len() - 1 {
            current = self.weights[i]
                .dot_multiply(&current)
                .add(&self.biases[i])
                .map(self.activation.function);
            self.data.push(current.clone());
        }
        current
    }

    /// Performs backpropagation to update the weights and biases of the neural network.
    ///
    /// # Arguments
    /// * `inputs` - A `Matrix` containing the input data for the network.
    /// * `targets` - A `Matrix` containing the target output data for the network.
    ///
    /// This function calculates the errors between the network's outputs and the target outputs,
    /// then uses those errors to update the weights and biases of the network through backpropagation.
    /// The learning rate is applied to the weight and bias updates.
    pub fn back_propogate(&mut self, inputs: Matrix, targets: Matrix) {
        let mut errors = targets.subtract(&inputs);
        let mut gradients = inputs.clone().map(self.activation.derivative);
        let map_with_learning_rate = |x: &f64| {
            let _lr = self.learning_rate;
            x * self.learning_rate
        };
        for i in (0..self.layers.len() - 1).rev() {
            gradients = gradients
                .elementwise_multiply(&errors)
                .map(map_with_learning_rate);
            self.weights[i] =
                self.weights[i].add(&gradients.dot_multiply(&self.data[i].transpose()));
            self.biases[i] = self.biases[i].add(&gradients);
            errors = self.weights[i].transpose().dot_multiply(&errors);
            gradients = self.data[i].map(self.activation.derivative);
        }
    }

    /// Trains the neural network by iterating through the provided input and target data for the specified number of epochs.
    ///
    /// # Arguments
    /// * `inputs` - A vector of input data matrices.
    /// * `targets` - A vector of target output data matrices.
    /// * `epochs` - The number of training epochs to perform.
    ///
    /// This function performs the following steps:
    /// 1. Iterates through the specified number of training epochs.
    /// 2. For each epoch, iterates through the input and target data.
    /// 3. For each input-target pair, performs a forward pass through the network using `feed_forward()` and then updates the weights and biases using `back_propogate()`.
    /// 4. Displays a progress bar to indicate the training progress.
    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: u32) {
        let bar = AvanceBar::new(epochs as u64);
        bar.set_desc("Progress");
        for i in 1..=epochs {
            if epochs < 100 || i % (epochs / 100) == 0 {
                // println!("Epoch {} of {}", i, epochs);
            }
            for j in 0..inputs.len() {
                let outputs = self.feed_forward(Matrix::from(inputs[j].clone()));
                self.back_propogate(outputs, Matrix::from(targets[j].clone()));
            }
            bar.inc();
        }
    }
}
