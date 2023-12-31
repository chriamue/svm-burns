use crate::{kernel::Kernel, svm::SVM};

pub struct SVC {
    pub kernel: Box<dyn Kernel>,
    pub w: Option<Vec<f64>>,                    // weight vector
    pub b: Option<f64>,                         // bias
    pub support_vectors: Option<Vec<Vec<f64>>>, // support vectors
}

impl SVC {
    pub fn new(kernel: Box<dyn Kernel>) -> SVC {
        SVC {
            kernel,
            w: None,
            b: None,
            support_vectors: None,
        }
    }

    pub fn get_model_parameters(&self) -> (Option<Vec<f64>>, Option<f64>) {
        (self.w.clone(), self.b)
    }
}

impl SVM for SVC {
    fn fit(&mut self, x: &Vec<Vec<f64>>, y: &Vec<i32>) {
        if x.len() != y.len() {
            panic!("Number of samples in x does not match number of labels in y");
        }

        let mut sum_positive = vec![0.0; x[0].len()];
        let mut sum_negative = vec![0.0; x[0].len()];
        let mut count_positive = 0;
        let mut count_negative = 0;

        for (i, sample) in x.iter().enumerate() {
            if y[i] > 0 {
                for j in 0..sample.len() {
                    sum_positive[j] += sample[j];
                }
                count_positive += 1;
            } else {
                for j in 0..sample.len() {
                    sum_negative[j] += sample[j];
                }
                count_negative += 1;
            }
        }

        let avg_positive: Vec<f64> = sum_positive
            .iter()
            .map(|&val| val / count_positive as f64)
            .collect();
        let avg_negative: Vec<f64> = sum_negative
            .iter()
            .map(|&val| val / count_negative as f64)
            .collect();

        self.w = Some(
            y.iter()
                .map(|&label| if label > 0 { 1.0 } else { -1.0 })
                .collect(),
        );

        self.b = Some(
            0.5 * (avg_positive.iter().sum::<f64>() - avg_negative.iter().sum::<f64>())
                / x[0].len() as f64,
        );

        self.support_vectors = Some(x.clone());
    }

    fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<i32> {
        let instances = self
            .support_vectors
            .as_ref()
            .expect("Model has not been trained yet.");
        let w = self.w.as_ref().expect("Model has not been trained yet.");
        let b = self.b.as_ref().expect("Model has not been trained yet.");
        x.iter()
            .map(|sample| {
                let mut decision = 0.0;
                for (support_vector, &weight) in instances.iter().zip(w.iter()) {
                    decision += weight * self.kernel.compute(sample, support_vector);
                }
                decision += b;

                if decision > 0.0 {
                    1
                } else {
                    -1
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{KernelType, RBFKernel};

    #[test]
    fn it_works() {
        let rbf = KernelType::rbf();
        let mut svc = SVC::new(rbf);

        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![-1, -1, 1];

        svc.fit(&x, &y);

        let predictions = svc.predict(&x);

        assert_eq!(predictions.len(), x.len());
    }

    mod tests {
        use super::*;

        #[test]
        fn svc_fit_predict() {
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

            let rbf_kernel = RBFKernel::new(1.0);

            let mut svc = SVC::new(Box::new(rbf_kernel));

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
    }
}
