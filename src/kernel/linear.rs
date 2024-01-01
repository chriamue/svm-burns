use serde::{Deserialize, Serialize};

use crate::Kernel;

#[derive(Default, Serialize, Deserialize)]
pub struct LinearKernel {}

impl LinearKernel {
    pub fn new() -> LinearKernel {
        LinearKernel {}
    }
}

impl Kernel for LinearKernel {
    fn compute(&self, x: &Vec<f64>, y: &Vec<f64>) -> f64 {
        x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum()
    }

    fn type_of(&self) -> super::KernelType {
        super::KernelType::Linear
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_kernel() {
        let v1 = vec![1., 2., 3.];
        let v2 = vec![4., 5., 6.];

        let result = LinearKernel::default().compute(&v1, &v2);

        assert_eq!(32.0, result);
    }
}
