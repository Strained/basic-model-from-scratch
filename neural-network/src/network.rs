use matrix::matrix::Matrix;
use crate::activations::Activation;
use avance::AvanceBar;


pub struct Network {
    layers: Vec<usize>,  // amount of neurons in each layer
    weights: Vec<Matrix>,  // weights of each neuron
    biases: Vec<Matrix>,  // biases of each neuron
    data: Vec<Matrix>,  // input data
    activation: Activation,  // activation function
    learning_rate: f64,   // learning rate
}
impl Network {
    pub fn new(layers: Vec<usize>, activation: Activation, desired_learning_rate: f64) -> Self {
        let mut weights: Vec<Matrix> = vec![];
        let mut biases: Vec<Matrix> = vec![];
        for i in 0..layers.len() - 1 {
            weights.push(Matrix::random(layers[i + 1], layers[i]));
            biases.push(Matrix::random(layers[i + 1], 1));
        }
        let default_learning_rate: f64 = 0.5; let learning_rate: f64;
        if default_learning_rate != desired_learning_rate {learning_rate = desired_learning_rate;}
        else {learning_rate = default_learning_rate;};
    return Network {layers, weights, biases, data: vec![], activation, learning_rate, /*map_with_learning_rate*/};}

    pub fn feed_forward(&mut self, inputs: Matrix) -> Matrix {
        assert!(self.layers[0] == inputs.data.len(), "Invalid Number of Inputs");
        //   println!("{:?} {:?}",self.weights[0],inputs);
        //   println!("{:?}",self.weights[0].dot_multiply(&inputs).add(&self.biases[0]));
        let mut current = inputs;
        self.data = vec![current.clone()];
        for i in 0..self.layers.len() -1 {
            current = self.weights[i]
            .dot_multiply(&current)
            .add(&self.biases[i]).map(self.activation.function);
            self.data.push(current.clone());
        }
    return current;}

    pub fn back_propogate(&mut self, inputs: Matrix, targets: Matrix) {
        let mut errors = targets.subtract(&inputs);
        let mut gradients = inputs.clone().map(self.activation.derivative);
        let map_with_learning_rate = |x: &f64| {let _lr = self.learning_rate; x * self.learning_rate};
        for i in (0..self.layers.len() - 1).rev() {
            gradients = gradients.elementwise_multiply(&errors).map(map_with_learning_rate);
            self.weights[i] = self.weights[i].add(&gradients.dot_multiply(&self.data[i].transpose()));
            self.biases[i] = self.biases[i].add(&gradients);
            errors = self.weights[i].transpose().dot_multiply(&errors);
            gradients = self.data[i].map(self.activation.derivative);
    }}
    
    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: u32) {
        let bar = AvanceBar::new(epochs as u64);
        bar.set_desc("Progress");
        for i in 1..=epochs {
            if epochs < 100 || i % (epochs / 100) == 0 {
                // println!("Epoch {} of {}", i, epochs);
            }
            for j in 0..inputs.len() {
                let outputs = self.feed_forward(Matrix::from(inputs[j].clone()));
                self.back_propogate(outputs,Matrix::from(targets[j].clone()));
        
    };bar.inc();};}
}