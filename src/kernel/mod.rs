// sources: https://github.com/smartcorelib/smartcore/blob/development/src/svm/mod.rs

pub mod linear;
pub mod rbf;

pub use linear::LinearKernel;
pub use rbf::RBFKernel;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub enum KernelType {
    Linear,
    RBF(f64),
}

impl KernelType {
    pub fn linear() -> Box<dyn Kernel> {
        Box::new(LinearKernel::default())
    }

    pub fn rbf() -> Box<dyn Kernel> {
        Box::new(RBFKernel::default())
    }

    pub fn new(&self) -> Box<dyn Kernel> {
        match self {
            KernelType::Linear => KernelType::linear(),
            KernelType::RBF(gamma) => Box::new(RBFKernel::new(*gamma)),
        }
    }
}

pub trait Kernel: Sync + Send {
    fn compute(&self, x: &Vec<f64>, y: &Vec<f64>) -> f64;
    fn type_of(&self) -> KernelType;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let rbf = KernelType::rbf();
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];
        let result = rbf.compute(&x, &y);
        assert_eq!(result, 1.0);
    }

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
