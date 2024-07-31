use rand::Rng;
use std::fmt;

#[derive(Debug, Clone)]
/// A matrix data structure that stores a 2D array of floating-point values.
///
/// The `Matrix` struct has three public fields:
/// - `rows`: the number of rows in the matrix
/// - `cols`: the number of columns in the matrix
/// - `data`: a `Vec<f64>` that stores the matrix elements in row-major order
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}
impl Matrix {
    // // Debugging function, helps to see the calling function, I used it to find where the matrices were created
    // pub fn get_caller_function_name() -> String {
    //     let mut s = String::new();
    //     s.push_str(&format!("Called from: "));
    //     s.push_str(&format!("{}:{} ", file!(), line!()));   // Get the file and line number of the caller function
    //     let module_name = std::module_path!();  // Get the name of the current module
    //     let caller_name = std::any::type_name::<Self>();    // Get the name of the caller function
    //     s.push_str(&format!("{}.{}", module_name, caller_name));    // Add current and caller modules to the string
    // return s;}  // Return the module name, caller module name, caller file and line number

    /// Performs element-wise multiplication between the current matrix and the provided matrix.
    ///
    /// # Arguments
    /// * `other` - The matrix to multiply element-wise with the current matrix.
    ///
    /// # Returns
    /// A new `Matrix` instance containing the result of the element-wise multiplication.
    ///
    /// # Panics
    /// Panics if the dimensions of the current matrix and the provided matrix do not match.
    pub fn elementwise_multiply(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Attempted to multiply by matrix of incorrect dimensions");
        }
        let mut result_data = vec![0.0; self.cols * self.rows];
        for (i, &value) in self.data.iter().enumerate() {
            result_data[i] = value * other.data[i]
        }
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: result_data,
        }
    }

    /// Creates a new `Matrix` instance the given number of rows and columns.
    /// Where each value is randomly generated.
    ///
    /// # Arguments
    /// * `rows` - The number of rows in the matrix.
    /// * `cols` - The number of columns in the matrix.
    ///
    /// # Returns
    /// A new `Matrix` instance with random values.
    pub fn random(rows: usize, cols: usize) -> Matrix {
        let mut buffer = Vec::<f64>::with_capacity(rows * cols);
        for _ in 0..buffer.capacity() {
            let num = rand::thread_rng().gen_range(0.0..1.0);
            buffer.push(num);
        }
        Matrix {
            rows,
            cols,
            data: buffer,
        }
    }

    /// Creates a new `Matrix` instance with the given number of rows, columns, and data.
    ///
    /// # Arguments
    /// * `rows` - The number of rows in the matrix.
    /// * `cols` - The number of columns in the matrix.
    /// * `data` - A vector containing the data for the matrix.
    ///
    /// # Panics
    /// Panics if the length of the `data` vector does not match the expected size of the matrix.
    ///
    /// # Returns
    /// A new `Matrix` instance with the specified rows, columns, and data.
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Matrix {
        assert!(data.len() - 1 != rows * cols, "Invalid Size");
        Matrix { rows, cols, data }
    }

    /// CCreates a new `Matrix` instance with the given number of rows and columns.
    /// Where all elements are initialized to 0.0.
    ///
    /// # Arguments
    /// * `rows` - The number of rows in the matrix.
    /// * `cols` - The number of columns in the matrix.
    ///
    /// # Returns
    /// A new `Matrix` instance with all elements set to 0.0.
    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        let buffer: Vec<f64> = vec![0.0; rows * cols];
        Matrix {
            rows,
            cols,
            data: buffer,
        }
    }

    /// Adds two matrices together and returns a new matrix with the result.
    ///
    /// # Arguments
    /// * `self` - The first matrix to add.
    /// * `other` - The second matrix to add.
    ///
    /// # Panics
    /// Panics if the dimensions of the two matrices do not match.
    ///
    /// # Returns
    /// A new `Matrix` instance with the result of adding the two input matrices.
    pub fn add(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Attempted to add matrix of incorrect dimensions");
        }
        let mut buffer: Vec<f64> = Vec::<f64>::with_capacity(self.rows * self.cols);
        for i in 0..self.data.len() {
            let result = self.data[i] + other.data[i];
            buffer.push(result);
        }
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: buffer,
        }
    }

    /// Subtracts two matrices and returns a new matrix with the result.
    ///
    /// # Arguments
    /// * `self` - The first matrix to subtract.
    /// * `other` - The second matrix to subtract.
    ///
    /// # Panics
    /// Panics if the dimensions of the two matrices do not match.
    ///
    /// # Returns
    /// A new `Matrix` instance with the result of subtracting the two input matrices.
    pub fn subtract(&self, other: &Matrix) -> Matrix {
        assert!(
            self.rows == other.rows && self.cols == other.cols,
            "Cannot subtract matrices with different dimensions"
        );
        let mut buffer: Vec<f64> = Vec::<f64>::with_capacity(self.rows * self.cols);
        for i in 0..self.data.len() {
            let result = self.data[i] - other.data[i];
            buffer.push(result);
        }
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: buffer,
        }
    }

    /// Computes the dot product of two matrices and returns a new matrix with the result.
    ///
    /// # Arguments
    /// * `self` - The first matrix.
    /// * `other` - The second matrix.
    ///
    /// # Panics
    /// Panics if the number of columns in the first matrix does not match the number of rows in the second matrix.
    ///
    /// # Returns
    /// A new `Matrix` instance with the result of the dot product of the two input matrices.
    pub fn dot_multiply(&self, other: &Matrix) -> Matrix {
        // // for debugging, print function name and args passed, the the call details:
        // log_vars!(self, other);
        // println!("{}",Self::get_caller_function_name());
        // // for debugging, print size of the self and other matrices
        // println!("Matrix Sizes - self.rows: {}, self.cols: {}, other.rows: {}, other.cols: {}", self.rows, self.cols, other.rows, other.cols);
        if self.cols != other.rows {
            panic!("Attempted to multiply by matrix of incorrect dimensions");
        }
        let mut result_data: Vec<f64> = vec![0.0; self.rows * other.cols];
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
                result_data[i * other.cols + j] = sum;
            }
        }
        Matrix {
            rows: self.rows,
            cols: other.cols,
            data: result_data,
        }
    }

    /// Computes the transpose of the matrix and returns a new matrix with the result.
    ///
    /// # Returns
    /// A new `Matrix` instance with the transpose of the input matrix.
    pub fn transpose(&self) -> Matrix {
        let mut buffer: Vec<f64> = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                buffer[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        Matrix {
            rows: self.cols,
            cols: self.rows,
            data: buffer,
        }
    }

    /// Applies the given function `func` to each element of the matrix and returns a new matrix with the transformed values.
    ///
    /// # Parameters
    /// - `func`: A closure that takes a reference to a `f64` value and returns a new `f64` value.
    ///
    /// # Returns
    /// A new `Matrix` instance with the same dimensions as the original matrix, but with each element transformed by the provided function.
    pub fn map<F>(&mut self, func: F) -> Matrix
    where
        F: Fn(&f64) -> f64,
    {
        let mut result = Matrix {
            rows: self.rows,
            cols: self.cols,
            data: Vec::with_capacity(self.data.len()),
        };
        result.data.extend(self.data.iter().map(|&val| func(&val)));
        result
    }
}

/// Converts a `Vec<f64>` into a `Matrix` with a single column.
///
/// # Parameters
/// - `vec`: The `Vec<f64>` to convert.
///
/// # Returns
/// A new `Matrix` instance with the same elements as the input `Vec<f64>`, and a single column.
impl From<Vec<f64>> for Matrix {
    fn from(vec: Vec<f64>) -> Self {
        let rows = vec.len();
        let cols = 1;
        Matrix {
            rows,
            cols,
            data: vec,
        }
    }
}

/// Implements the `PartialEq` trait for the `Matrix` struct, allowing two `Matrix` instances to be compared for equality.
///
/// The `eq` method compares the dimensions (rows and columns) of the two matrices. 
/// If the dimensions are not equal, the matrices are considered not equal. 
/// If the dimensions are equal, the method compares each element of the matrices. 
/// If any element is not equal, the matrices are considered not equal. 
/// If all elements are equal, the matrices are considered equal.
impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }
        for i in 0..self.data.len() {
            if self.data[i] != other.data[i] {
                return false;
            }
        }
        true // Only if expected elements are equal
    }
}

/// Implements the `fmt::Display` trait for the `Matrix` struct, allowing a `Matrix` instance to be printed using the `println!` macro 
/// or other formatting functions.
///
/// The `fmt` method iterates over the rows and columns of the matrix, writing each element to the provided `fmt::Formatter`. 
/// Columns are separated by a tab character, and rows are separated by a newline character.
impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in 0..self.rows {
            for col in 0..self.cols {
                write!(f, "{}", self.data[row * self.cols + col])?;
                if col < self.cols - 1 {
                    write!(f, "\t")?; // Separate columns with a tab
                }
            }
            writeln!(f)?; // Move to the next line after processing all columns in the row
        }
        Ok(())
    }
}

#[cfg(test)]
/// This module contains unit tests for the `Matrix` struct and its associated methods.
///
/// The tests cover various functionality of the `Matrix` struct, including:
/// - Creating a random matrix with specified dimensions
/// - Performing element-wise multiplication between two matrices
/// - Subtracting two matrices with the same dimensions
/// - Performing dot product multiplication between two matrices
/// - Subtracting two matrices with different dimensions (expected to panic)
/// - Adding two matrices
/// - Transposing matrices of different dimensions
/// - Applying a mapping function to each element of a matrix
mod tests {
    use super::*;
    use crate::matrix;

    #[test]
    fn test_random_matrix() {
        let rows = 3;
        let cols = 4;
        let matrix = Matrix::random(rows, cols);
        assert_eq!(matrix.rows, rows);
        assert_eq!(matrix.cols, cols);
        assert_eq!(matrix.data.len(), rows * cols);
        for &num in &matrix.data {
            assert!((0.0..1.0).contains(&num));
        }
    }
    #[test]
    fn test_elementwise_multiply() {
        // Create two matrices for testing
        let matrix1 = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let matrix2 = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let result = matrix1.elementwise_multiply(&matrix2); // Perform element-wise multiplication
        let expected_result = Matrix::new(2, 2, vec![5.0, 12.0, 21.0, 32.0]); // Define the expected result
        assert_eq!(result, expected_result); // Check if the actual result matches the expected result
    }
    #[test]
    fn test_subtract_same_dimensions() {
        let matrix1 = matrix![1.0, 2.0;
                              3.0, 4.0];
        let matrix2 = matrix![5.0, 6.0;
                              7.0, 8.0];
        let result = matrix1.subtract(&matrix2);
        let expected = matrix![-4.0, -4.0;
                               -4.0, -4.0];
        assert_eq!(result, expected);
    }
    #[test]
    fn test_dot_multiply() {
        let a = matrix![1.0, 2.0, 3.0;
                        4.0, 5.0, 6.0];
        let b = matrix![7.0, 8.0;
                        9.0, 10.0;
                        11.0, 12.0];
        let result = a.dot_multiply(&b);
        let expected_result = matrix![58.0, 64.0;
                                      139.0, 154.0];
        assert_eq!(result, expected_result);
    }
    #[test]
    #[should_panic(expected = "Cannot subtract matrices with different dimensions")]
    fn test_subtract_different_dimensions() {
        let matrix1 = matrix![1.0, 2.0;
                              3.0, 4.0];
        let matrix2 = matrix![5.0, 6.0, 7.0;
                              8.0, 9.0, 10.0];
        let _ = matrix1.subtract(&matrix2);
    }
    #[test]
    fn test_matrix_addition() {
        let a = matrix![1.0, 2.0, 3.0;
                        4.0, 5.0, 6.0;
                        7.0, 8.0, 9.0];
        let b = matrix![5.0, 6.0, 7.0;
                        8.0, 9.0, 10.0;
                        11.0, 12.0, 13.0];
        let expected_result = matrix![6.0, 8.0, 10.0;
                                      12.0, 14.0, 16.0;
                                      18.0, 20.0, 22.0];
        let result = a.add(&b);
        assert_eq!(result, expected_result);
    }
    #[test]
    fn test_transpose_2x2() {
        let matrix = matrix![1.0, 2.0;
                             3.0, 4.0];
        let transposed = matrix.transpose();
        let expected = matrix![1.0, 3.0;
                               2.0, 4.0];
        assert_eq!(transposed, expected);
    }
    #[test]
    fn test_transpose_3x3() {
        let matrix = matrix![1.0, 2.0, 3.0;
                             4.0, 5.0, 6.0;
                             7.0, 8.0, 9.0];
        let transposed = matrix.transpose();
        let expected = matrix![1.0, 4.0, 7.0;
                               2.0, 5.0, 8.0;
                               3.0, 6.0, 9.0];
        assert_eq!(transposed, expected);
    }
    #[test]
    fn test_transpose_4x3() {
        let matrix = matrix![1.0, 2.0, 3.0;
                             4.0, 5.0, 6.0;
                             7.0, 8.0, 9.0;
                             10.0, 11.0, 12.0];
        let transposed = matrix.transpose();
        let expected = matrix![1.0, 4.0, 7.0, 10.0;
                               2.0, 5.0, 8.0, 11.0;
                               3.0, 6.0, 9.0, 12.0];
        assert_eq!(transposed, expected);
    }
    #[test]
    fn test_map_add_one() {
        let mut matrix = Matrix {
            rows: 2,
            cols: 2,
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let transformed = matrix.map(|x| x + 1.0);
        let expected = Matrix {
            rows: 2,
            cols: 2,
            data: vec![2.0, 3.0, 4.0, 5.0],
        };
        assert_eq!(transformed, expected);
    }
    #[test]
    fn test_map_square() {
        let mut matrix = Matrix {
            rows: 2,
            cols: 2,
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let transformed = matrix.map(|x| x * x);
        let expected = Matrix {
            rows: 2,
            cols: 2,
            data: vec![1.0, 4.0, 9.0, 16.0],
        };
        assert_eq!(transformed, expected);
    }
}
