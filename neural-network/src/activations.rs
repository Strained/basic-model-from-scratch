use std::f64::consts::E;

#[derive(Clone,Copy,Debug)]
/// An activation function and its derivative.
///
/// Activation functions are used in neural networks to introduce non-linearity
/// into the model. The `function` field is the activation function itself, and
/// the `derivative` field is the derivative of the activation function, which
/// is used during backpropagation.
pub struct Activation {
    /// The activation function.
    pub function: fn(&f64) -> f64,
    /// The derivative of the activation function.
    pub derivative: fn(&f64) -> f64,
}

/// The sigmoid activation function and its derivative.
///
/// The sigmoid function is a common activation function used in neural networks.
/// It maps any input value to a value between 0 and 1, making it useful for
/// binary classification problems. The derivative of the sigmoid function is
/// also provided, which is used during backpropagation.
pub const SIGMOID: Activation = Activation {
    function: |x| 1.0 / (1.0 + E.powf(-x)),
    derivative: |x| x * (1.0 - x),
};