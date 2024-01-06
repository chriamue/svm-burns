#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::{
    kernel::Kernel, optimizer::Optimizer, parameters::Parameters, smo::SMO, svm::SVM, B, W,
};

#[derive(Serialize, Deserialize)]
pub struct SVC {
    parameters: Parameters,
    alphas: Option<Vec<f64>>,
    support_vectors: Option<Vec<Vec<f64>>>,
    support_labels: Option<Vec<f64>>,
    w: Option<W>,
    b: Option<B>,
}

unsafe impl Sync for SVC {}
unsafe impl Send for SVC {}

impl SVC {
    pub fn new(parameters: Parameters) -> SVC {
        SVC {
            parameters,
            alphas: None,
            support_vectors: None,
            support_labels: None,
            w: None,
            b: None,
        }
    }

    pub fn predict_row(
        x_i: &Vec<f64>,
        w: &[f64],
        support_vectors: &[Vec<f64>],
        b: f64,
        kernel: &Box<dyn Kernel>,
    ) -> f64 {
        let mut sum = b;
        for (w_i, support_vector) in w.iter().zip(support_vectors) {
            sum += w_i * kernel.compute(x_i, support_vector);
        }
        sum
    }

    pub fn decision_function(&self, x: &Vec<Vec<f64>>) -> Vec<f64> {
        let support_vectors = self.support_vectors.as_ref().expect("Model not trained");
        let w = self.w.as_ref().expect("Model not trained");
        let b = self.b.expect("Model not trained");

        let y: Vec<f64> = x
            .iter()
            .map(|sample| Self::predict_row(sample, w, support_vectors, b, &self.parameters.kernel))
            .collect();
        y
    }
}
impl SVM for SVC {
    fn fit(&mut self, x: &Vec<Vec<f64>>, y: &Vec<i32>) {
        if x.len() != y.len() {
            panic!("Number of samples in x does not match number of labels in y");
        }

        let y_f64: Vec<f64> = y.iter().map(|&label| label as f64).collect();

        let smo = SMO::new(
            self.parameters.c,
            self.parameters.tol,
            self.parameters.epochs,
        );
        let (alphas, b) = smo.optimize(x, &y_f64, &self.parameters.kernel);

        let mut support_vectors = Vec::new();
        let mut support_labels = Vec::new();

        for (i, &alpha) in alphas.iter().enumerate() {
            if alpha.abs() > 1e-5 {
                support_vectors.push(x[i].clone());
                support_labels.push(y_f64[i]);
            }
        }

        self.w = Some(alphas);
        self.support_vectors = Some(support_vectors);
        self.support_labels = Some(support_labels);
        self.b = Some(b);
    }

    fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<i32> {
        let y_hat = self.decision_function(x);
        y_hat.iter()
            .map(|&y| if y > 0.0 { 1 } else { -1 })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use rayon::{result, vec};

    use super::*;
    use crate::{
        kernel::{KernelType, LinearKernel, RBFKernel},
        parameters, support_vector,
    };

    #[test]
    fn it_works() {
        let mut parameters = Parameters::default();
        parameters.with_kernel(KernelType::linear());
        let mut svc = SVC::new(parameters);

        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![-1, -1, 1];

        svc.fit(&x, &y);

        let predictions = svc.predict(&x);

        assert_eq!(predictions.len(), x.len());
    }

    #[test]
    fn svc_fit_predict_rbf() {
        let x = vec![
            vec![5.1, 3.5, 1.4, 0.2],
            vec![4.9, 3.0, 1.4, 0.2],
            vec![4.7, 3.2, 1.3, 0.2],
            vec![4.6, 3.1, 1.5, 0.2],
            vec![5.0, 3.6, 1.4, 0.2],
            vec![5.4, 3.9, 1.7, 0.4],
            vec![4.6, 3.4, 1.4, 0.3],
            vec![5.0, 3.4, 1.5, 0.2],
            vec![4.4, 2.9, 1.4, 0.2],
            vec![4.9, 3.1, 1.5, 0.1],
            vec![7.0, 3.2, 4.7, 1.4],
            vec![6.4, 3.2, 4.5, 1.5],
            vec![6.9, 3.1, 4.9, 1.5],
            vec![5.5, 2.3, 4.0, 1.3],
            vec![6.5, 2.8, 4.6, 1.5],
            vec![5.7, 2.8, 4.5, 1.3],
            vec![6.3, 3.3, 4.7, 1.6],
            vec![4.9, 2.4, 3.3, 1.0],
            vec![6.6, 2.9, 4.6, 1.3],
            vec![5.2, 2.7, 3.9, 1.4],
        ];
        let y: Vec<i32> = vec![
            -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ];

        let mut parameters = Parameters::default();
        parameters.with_kernel(Box::new(RBFKernel::new(0.7)));
        let mut svc = SVC::new(parameters);

        svc.fit(&x, &y);

        let predictions = svc.predict(&x);

        let correct_predictions = predictions
            .iter()
            .zip(y.iter())
            .filter(|(&pred, &true_label)| (pred > 0) == (true_label > 0))
            .count();
        let accuracy = correct_predictions as f64 / x.len() as f64;

        assert!(
            accuracy >= 0.9,
            "Accuracy ({accuracy}) is not larger or equal to 0.9"
        );
    }

    #[test]
    fn svc_fit_predict_linear() {
        let x = vec![
            vec![5.1, 3.5, 1.4, 0.2],
            vec![4.9, 3.0, 1.4, 0.2],
            vec![4.7, 3.2, 1.3, 0.2],
            vec![4.6, 3.1, 1.5, 0.2],
            vec![5.0, 3.6, 1.4, 0.2],
            vec![5.4, 3.9, 1.7, 0.4],
            vec![4.6, 3.4, 1.4, 0.3],
            vec![5.0, 3.4, 1.5, 0.2],
            vec![4.4, 2.9, 1.4, 0.2],
            vec![4.9, 3.1, 1.5, 0.1],
            vec![7.0, 3.2, 4.7, 1.4],
            vec![6.4, 3.2, 4.5, 1.5],
            vec![6.9, 3.1, 4.9, 1.5],
            vec![5.5, 2.3, 4.0, 1.3],
            vec![6.5, 2.8, 4.6, 1.5],
            vec![5.7, 2.8, 4.5, 1.3],
            vec![6.3, 3.3, 4.7, 1.6],
            vec![4.9, 2.4, 3.3, 1.0],
            vec![6.6, 2.9, 4.6, 1.3],
            vec![5.2, 2.7, 3.9, 1.4],
        ];
        let y: Vec<i32> = vec![
            -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ];

        let linear_kernel = LinearKernel::default();

        let mut parameters = Parameters::default();
        parameters.with_kernel(Box::new(linear_kernel));

        let mut svc = SVC::new(parameters);

        svc.fit(&x, &y);

        let predictions = svc.predict(&x);

        let correct_predictions = predictions
            .iter()
            .zip(y.iter())
            .filter(|(&pred, &true_label)| (pred > 0) == (true_label > 0))
            .count();
        let accuracy = correct_predictions as f64 / x.len() as f64;

        assert!(
            accuracy >= 0.9,
            "Accuracy ({accuracy}) is not larger or equal to 0.9"
        );
    }

    #[test]
    fn test_predict_w_b() {
        let x = vec![vec![1.0, 1.0], vec![2.0, 2.0], vec![3.0, 3.0]];
        let y: Vec<i32> = vec![-1, -1, 1];
        let support_vectors = vec![vec![1.0, 1.0], vec![2.0, 2.0]];

        let w = vec![1.0];
        let b = 0.0;

        let kernel: Box<dyn Kernel> = Box::new(LinearKernel::default());

        let result = SVC::predict_row(&x[0], &w, &support_vectors, b, &kernel);
        assert_eq!(result, 2.0);

        let result = SVC::predict_row(&x[1], &w, &support_vectors, b, &kernel);
        assert_eq!(result, 4.0);
    }

    #[test]
    fn test_decision_function() {
        let x = vec![vec![1.0, 1.0], vec![2.0, 2.0], vec![3.0, 3.0]];
        let support_vectors = vec![vec![1.0, 1.0], vec![2.0, 2.0]];

        let w = vec![1.0];
        let b = 0.0;

        let kernel: Box<dyn Kernel> = Box::new(LinearKernel::default());

        let mut parameters = Parameters::default();
        parameters.with_kernel(kernel);

        let svc = SVC {
            parameters,
            alphas: None,
            support_vectors: Some(support_vectors),
            support_labels: None,
            w: Some(w),
            b: Some(b),
        };

        let result = svc.decision_function(&x);
        assert_eq!(result, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_predict() {
        let x = vec![vec![1.0, 1.0], vec![2.0, 2.0], vec![3.0, 3.0]];
        let support_vectors = vec![vec![1.0, 1.0], vec![2.0, 2.0]];

        let w = vec![1.0];
        let b = 0.0;

        let kernel: Box<dyn Kernel> = Box::new(LinearKernel::default());

        let mut parameters = Parameters::default();
        parameters.with_kernel(kernel);

        let svc = SVC {
            parameters,
            alphas: None,
            support_vectors: Some(support_vectors),
            support_labels: None,
            w: Some(w),
            b: Some(b),
        };

        let result = svc.predict(&x);
        assert_eq!(result, vec![1, 1, 1]);
    }
}
