#[allow(unused_imports)]    // Required for the compiler to know what Matrix is from this file 
use crate::matrix::Matrix;  // context, but not directly used, so results in a compiler warning

#[macro_export]
macro_rules! matrix {
    ( $( $($val:expr),+ );* $(;)? ) => {
        {
            let mut data = Vec::<f64>::new();
            let mut rows = 0;
            let mut cols = 0;
            $(
                let row_data = vec![$($val),+];
                data.extend(row_data);
                rows += 1;
                let row_len = vec![$($val),+].len();
                if cols == 0 {
                    cols = row_len;
                } else if cols != row_len {
                    panic!("Inconsistent number of elements in the matrix rows");
                }
            )*

            Matrix {
                rows,
                cols,
                data,
            }
        }
    };
}


#[macro_export] 
// For debugging: Returns parameters passed to the function
macro_rules! log_vars {
    ($($arg:ident),*) => {{
        println!("At function: {} with variables: {:?}", $crate::function!(true), vec![$(stringify!($arg)),*]);
    }};
}

#[macro_export] 
// For debugging: Returns the function name w/ or w/o the path based on input bool
macro_rules! function {
    ($val:expr $(,)?) => {match $val {
        false => {
            fn f() {}
            fn type_name_of<T>(_: T) -> &'static str {
                std::any::type_name::<T>()
            }
            type_name_of(f)
            .rsplit("::")
            .find(|&part| part != "f" && part != "{{closure}}")
            .expect("Short function name")
        }
        true => {
            fn f() {}
            fn type_name_of<T>(_: T) -> &'static str {
                std::any::type_name::<T>()
            }
            let name = type_name_of(f);
            name.strip_suffix("::f").unwrap()
        }
    }};
}


#[cfg(test)]
/// Tests the `matrix!` macro by creating a 3x3 matrix and asserting its properties.
///
/// This test verifies that the `matrix!` macro correctly creates a `Matrix` struct 
/// with the expected number of rows and columns, and the expected data values.
mod tests {
    use super::Matrix;

    #[test]
    fn test_matrix_macro() {
        let m = matrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0;
            7.0, 8.0, 9.0
        ];
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 3);
        assert_eq!(
            m.data,
            vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0,
            ]
        );
    }
}
