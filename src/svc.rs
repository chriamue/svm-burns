use crate::{kernel::Kernel, svm::SVM};

pub struct SVC {
    pub kernel: Box<dyn Kernel>,
}

impl SVC {
    pub fn new(kernel: Box<dyn Kernel>) -> SVC {
        SVC { kernel }
    }
}

impl SVM for SVC {
    fn fit(&mut self, x: &Vec<Vec<f64>>, y: &Vec<f64>) {
        if x.len() != y.len() {
            panic!("Number of samples in x does not match number of labels in y");
        }
    }

    fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<f64> {
        x.iter()
            .map(|sample| {
                let mut sum = 0.0;
                for i in 0..x.len() {
                    sum += self.kernel.compute(&sample, &x[i]);
                }
                sum
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::RBFKernel;

    #[test]
    fn it_works() {
        let rbf = RBFKernel::new(1.0);
        let mut svc = SVC::new(Box::new(rbf));

        // Example data
        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![-1.0, -1.0, 1.0];

        svc.fit(&x, &y);

        let predictions = svc.predict(&x);

        assert_eq!(predictions.len(), x.len());
    }
}
