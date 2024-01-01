use crate::{kernel, Kernel};

pub struct Parameters {
    /// Kernel
    pub kernel: Box<dyn Kernel>,
    /// regularization parameter
    pub c: f64,
    /// numerical tolerance
    pub tol: f64,
    /// maximum number of iterations over Larange multipliers without changing
    pub epochs: usize,
}

impl Parameters {
    pub fn new(kernel: Box<dyn Kernel>, c: f64, tol: f64, epochs: usize) -> Self {
        Parameters {
            kernel,
            c,
            tol,
            epochs,
        }
    }
}

impl Default for Parameters {
    fn default() -> Self {
        let kernel = kernel::RBFKernel::new(1.0);
        Parameters {
            kernel: Box::new(kernel),
            c: 1.0,
            tol: 1e-3,
            epochs: 5,
        }
    }
}

impl Parameters {
    pub fn with_kernel(&mut self, kernel: Box<dyn Kernel>) -> &mut Self {
        self.kernel = kernel;
        self
    }

    pub fn with_c(&mut self, c: f64) -> &mut Self {
        self.c = c;
        self
    }

    pub fn with_tol(&mut self, tol: f64) -> &mut Self {
        self.tol = tol;
        self
    }

    pub fn with_max_passes(&mut self, max_passes: usize) -> &mut Self {
        self.epochs = max_passes;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameters() {
        let mut params = Parameters::default();
        assert_eq!(params.c, 1.0);
        assert_eq!(params.tol, 1e-3);
        assert_eq!(params.epochs, 5);

        params.with_c(2.0).with_tol(2e-3).with_max_passes(10);
        assert_eq!(params.c, 2.0);
        assert_eq!(params.tol, 2e-3);
        assert_eq!(params.epochs, 10);
    }
}
