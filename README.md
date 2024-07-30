# Basic Model From Scratch

A Rust-based neural network framework implementing a basic XOR function intended as a learning exercise.

- This neural network trains to learn the XOR function.
- The inputs represent the two binary inputs
- The outputs are the XOR of those two inputs.

## Source Material - Attribution

This project is based on codemoonsxyz's educational project.  I am grateful for their contribution to the community and my education.  You can find them here:

- codemoonsxyz's GitHub page: (<https://github.com/codemoonsxyz>)
- Original project GitHub repo: <https://github.com/codemoonsxyz/neural-net-rs/tree/main>
- codemoonxyz's educational video: (<https://www.youtube.com/watch?v=DKbz9pNXVdE>)

## Features

**Simplicity:** Since this is a simple XOR function, it is easy to understand, and I can play around with the inputs, weights, and biases to see what happens.

**Flexibility:** The neural network is flexible by design, allowing experimentation with different neural network architectures.

## Getting Started

### Prerequisites

- **Install Rust:** (<https://www.rust-lang.org/learn/get-started>)

- **Setup Debugging** This is specific to your OS and IDE, but I use VS Code with the [Rust Extension Pack](https://marketplace.visualstudio.com/items?itemName=swellaby.rust-pack).

### Installation

    git clone https://github.com/Strained/basic-model-from-scratch.git
    cd basic-model-from-stratch
    cargo build
    cargo run

### Usage

#### Training: Cost Function and Back Propagation

> Train the neural network to learn the XOR function.

    cargo run -- --train

#### Testing: Forward Propagation

> Test the neural network to see if it can predict the XOR function.

    cargo run -- --forward

> If we only pass the `--forward` parameter the neural network will not be trained, so it will not be able to predict the XOR function better than 50/50 chance.

#### Testing: Training and Forward Propagation

> Train the neural network and then test it to see if it can predict the XOR function.

    cargo run -- --train --forward

## XOR Explained

> XOR may be novel for some, so I'm covering it here.

XOR (e***X***clusive ***OR***) is a logical operation that outputs true if the two inputs are different, and false if they are the same.

For example:

    Input 1 = 0, Input 2 = 0, Output = 0 (false)
    Input 1 = 0, Input 2 = 1, Output = 1 (true)
    Input 1 = 1, Input 2 = 0, Output = 1 (true)
    Input 1 = 1, Input 2 = 1, Output = 0 (false)

XOR returns true given two different inputs and false given two same inputs.
Unlike regular OR, which returns true if either input is true.

Some key properties of XOR:

- XOR(0, 0) = 0
- XOR(0, 1) = 1
- XOR(1, 0) = 1
- XOR(1, 1) = 0
- It is commutative - XOR(A, B) = XOR(B, A)
- It is associative - XOR(A, XOR(B, C)) = XOR(A, B, C)

---
