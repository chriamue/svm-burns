use std::ops::Mul;

use serde::{Deserialize, Serialize};

use crate::Kernel;

#[derive(Serialize, Deserialize)]
pub struct RBFKernel {
    gamma: f64,
}

impl Default for RBFKernel {
    fn default() -> Self {
        RBFKernel { gamma: 1.0 }
    }
}

impl RBFKernel {
    pub fn new(gamma: f64) -> RBFKernel {
        RBFKernel { gamma }
    }

    pub fn with_gamma(mut self, gamma: f64) -> RBFKernel {
        self.gamma = gamma;
        self
    }
}

impl Kernel for RBFKernel {
    fn compute(&self, x: &Vec<f64>, y: &Vec<f64>) -> f64 {
        x.iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - yi).powi(2))
            .sum::<f64>()
            .mul(-self.gamma)
            .exp()
    }

    fn type_of(&self) -> super::KernelType {
        super::KernelType::RBF(self.gamma)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rbf_kernel() {
        let v1 = vec![1., 2., 3.];
        let v2 = vec![4., 5., 6.];

        let result = RBFKernel::default()
            .with_gamma(0.055)
            .compute(&v1, &v2)
            .abs();

        assert!((0.2265f64 - result) < 1e-4);
    }
}
