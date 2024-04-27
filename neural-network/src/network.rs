use matrix::matrix::Matrix;
use crate::activations::Activation;

#[derive(Builder)]
pub struct Network {
    layers: Vec<usize>,  // amount of neurons in each layer
    weights: Vec<Matrix>,  // weights of each neuron
    biases: Vex<Matrix>,  // biases of each neuron
    data: Vec<Matrix>,  // input data
    activation: Activation,  // activation function
    learning_rate: f64,  // learning rate
}
impl Network {
    pub fn new(layers: Vec<usize>, activation: Activation, learning_rate: f64) -> Self {
        let mut weights: Vec<Matrix> = Vec![];
        let mut biases: Vec<Matrix> = Vec![];
        for i in 0..layers.len() - 1 {
            weights.push(Matrix::random(layers[i + 1], layers[1]));
            biases.push(Matrix::random(layers[i + 1], 1));
        }
        Network {
            layers,
            weights,
            biases,
            data: vec![],
            activation,
            learning_rate,   
        }
    }
    pub fn feed_forward(&mut self, inputs: Matrix) -> Matrix {
        assert!(self.layers[0] == inputs.data.len(), "Invalid number of inputs");
        let mut current = inputs;
        self.data = vec![current.clone()];
        for i in 0..self.layers.len() -1 {
            current = self.weights[i]
            .dot_multiply(&current)
            .add(&self.biases[i].map(self.activation.function));
            self.data.push(current.clone());
        }
        current
    }
    pub fn back_propagate(&mut self, inputs: Matrix, targets: Matrix) {
        let mut errors = targets.subtract(&inputs);
        let mut gradients = inputs.clone().map(self.activation.derivative);
        for i in (0..self.layers.len() - 1).rev() {
            gradients = gradients.elementwise_multiply(&errors).map(|x| x * self.learning_rate);     // In example this was * 0.5, but I believe it should be the learning rate value
            self.weights[i] = self.weights[i].add(&gradients.dot_multiply(&self.data[i].transpose()));
            self.biases[i] = self.biases[i].add(&gradients);
            errors = self.weights[i].transpose().dot_multiply(&errors);
            gradients = self.data[i].map(self.activation.derivative);
        }
    }
    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: u32) {
        for _ in 1..=epochs {
            if epochs < 100 || i % (epochs / 100) == 0 {
             println!("Epoch {} of {}", i, epochs);
            }
            for j in 0..inputs.len() {
                let outputs = self.feed_forward(Matrix::from(inputs[j].clone()));
                self.back_propagate(outputs, Matrix::from(targets[j].clone()));
            }
        }
    }
}