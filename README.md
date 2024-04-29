# basic-model-from-stractch
A Rust-based neural network framework implementing a basic XOR function intended as a learning exercise. 

- This neural network trains to learn the XOR function. 
- The inputs represent the two binary inputs
- The outputs are the XOR of those two inputs.

## Source Material - Attribution
This project is based on codemoonsxyz's educational project.  I am grateful for their contribution to the community and my education.  You can find them here:
- codemoonsxyz's github page: (https://github.com/codemoonsxyz)
- Original project github repo: https://github.com/codemoonsxyz/neural-net-rs/tree/main
- codemoonxyz's educational video: (https://www.youtube.com/watch?v=DKbz9pNXVdE)

## Features
**Simplicity:** Since this is a simple XOR function, it is easy to understand and I can play around with the inputs, weights, and biases to see what happens.

**Flexibility:** The neural network is flexible by design, allowing experimentation with different neural network architectures.


## Getting Started
### Prerequisites
- **Install Rust:** (https://www.rust-lang.org/learn/get-started)

- **Setup Debugging** This is specific to your OS and IDE, but I use VSCode with the [Rust Extension Pack](https://marketplace.visualstudio.com/items?itemName=swellaby.rust-pack).

### Installation

```bash
git clone https://github.com/Strained/basic-model-from-scratch.git
cd basic-model-from-stratch
cargo build
cargo run
```

## XOR Explained
**I wish I had understood this a little better when I started, so in case this concept is a bit new for you too:**

XOR (e***X***clusive ***OR***) is a logical operation that outputs true if the two inputs are different, and false if they are the same.

For example:
    Input 1 = 0, Input 2 = 0, Output = 0 (false)
    Input 1 = 0, Input 2 = 1, Output = 1 (true)
    Input 1 = 1, Input 2 = 0, Output = 1 (true)
    Input 1 = 1, Input 2 = 1, Output = 0 (false)

XOR returned true given two different inputs and false given two same inputs. Unlike regular OR, which returns true if either input is true.

Some key properties of XOR:
* XOR(0, 0) = 0
* XOR(0, 1) = 1
* XOR(1, 0) = 1
* XOR(1, 1) = 0
* It is commutative - XOR(A, B) = XOR(B, A)
* It is associative - XOR(A, XOR(B, C)) = XOR(A, B, C)
---