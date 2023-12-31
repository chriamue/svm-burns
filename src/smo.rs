// source: https://chubakbidpaa.com/svm/2020/12/27/smo-algorithm-simplifed-copy.html
// source: https://github.com/smartcorelib/smartcore/blob/development/src/svm/svc.rs

use rand::Rng;

use crate::{kernel::KernelType, Kernel};

pub struct SMO {
    /// regularization parameter
    c: f64,
    /// numerical tolerance
    tol: f64,
    /// maximum number of iterations over Larange multipliers without changing
    max_passes: usize,
    /// Kernel
    kernel: Box<dyn Kernel>,
}

impl SMO {
    pub fn new(c: f64, tol: f64, max_passes: usize, kernel: Box<dyn Kernel>) -> Self {
        SMO {
            c,
            tol,
            max_passes,
            kernel,
        }
    }
}

impl SMO {
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
        self.max_passes = max_passes;
        self
    }
}

impl Default for SMO {
    fn default() -> Self {
        let kernel = KernelType::linear();
        SMO::new(1.0, 1e-3, 5, kernel)
    }
}

impl SMO {
    fn linear_classifier(
        kernel: &Box<dyn Kernel>,
        alpha: &Vec<f64>,
        b: f64,
        x: &Vec<Vec<f64>>,
        y: &Vec<f64>,
    ) -> f64 {
        let mut sum = 0.0;
        for i in 0..x.len() {
            sum += alpha[i] * y[i] * kernel.compute(&x[i], &x[i]);
        }
        sum + b
    }

    /// Computes the lower bounds on Lagrange multipliers
    #[allow(non_snake_case)]
    fn compute_L(alpha_i: f64, alpha_j: f64, y_i: f64, y_j: f64, c: f64) -> f64 {
        if y_i != y_j {
            return f64::max(0.0, alpha_j - alpha_i);
        } else {
            return f64::max(0.0, alpha_i + alpha_j - c);
        }
    }

    /// Computes the upper bounds on Lagrange multipliers
    #[allow(non_snake_case)]
    fn compute_H(alpha_i: f64, alpha_j: f64, y_i: f64, y_j: f64, c: f64) -> f64 {
        if y_i != y_j {
            return f64::min(c, c + alpha_j - alpha_i);
        } else {
            return f64::min(c, alpha_i + alpha_j);
        }
    }

    /// Computes Eta
    /// Eta is the second derivative of the objective function
    fn calculate_eta(kernel: &Box<dyn Kernel>, x_i: &Vec<f64>, x_j: &Vec<f64>) -> f64 {
        2.0 * kernel.compute(x_i, x_j) - kernel.compute(x_i, x_i) - kernel.compute(x_j, x_j)
    }

    /// calculates the error for a given sample
    fn calculate_error(
        kernel: &Box<dyn Kernel>,
        x_i: &Vec<f64>,
        y_i: f64,
        b: f64,
        alpha: &Vec<f64>,
        x: &Vec<Vec<f64>>,
        y: &Vec<f64>,
    ) -> f64 {
        let f_x_i = SMO::linear_classifier(kernel, alpha, b, x, y);
        let e_i = f_x_i - y_i;
        e_i
    }

    fn clip_alpha_j(alpha_j: f64, l: f64, h: f64) -> f64 {
        if alpha_j > h {
            return h;
        } else if alpha_j < l {
            return l;
        } else {
            return alpha_j;
        }
    }

    fn calculate_alpha_j(alpha_j: f64, y_j: f64, e_i: f64, e_j: f64, eta: f64) -> f64 {
        alpha_j - y_j * (e_i - e_j) / eta
    }

    fn calculate_alpha_i(
        alpha_i: f64,
        y_i: f64,
        y_j: f64,
        alpha_j_old: f64,
        alpha_j_new: f64,
    ) -> f64 {
        alpha_i + y_i * y_j * (alpha_j_old - alpha_j_new)
    }

    fn calculate_bs(
        b: f64,
        x_i: &Vec<f64>,
        y_i: f64,
        x_j: &Vec<f64>,
        y_j: f64,
        alpha_i: f64,
        alpha_j: f64,
        e_i: f64,
        e_j: f64,
        alpha_i_old: f64,
        alpha_j_old: f64,
        kernel: &Box<dyn Kernel>,
    ) -> (f64, f64) {
        let b1 = b
            - e_i
            - y_i * (alpha_i - alpha_i_old) * kernel.compute(x_i, x_i)
            - y_j * (alpha_j - alpha_j_old) * kernel.compute(x_i, x_j);
        let b2 = b
            - e_j
            - y_i * (alpha_i - alpha_i_old) * kernel.compute(x_i, x_i)
            - y_j * (alpha_j - alpha_j_old) * kernel.compute(x_i, x_j);
        (b1, b2)
    }

    fn compute_b(b_one: f64, b_two: f64, alpha_i: f64, alpha_j: f64, c: f64) -> f64 {
        if 0.0 < alpha_i && alpha_i < c {
            return b_one;
        } else if 0.0 < alpha_j && alpha_j < c {
            return b_two;
        } else {
            return (b_one + b_two) / 2.0;
        }
    }

    fn rand_j(m: usize, i: usize) -> usize {
        let mut j = i;
        while j == i {
            j = rand::thread_rng().gen_range(0..m);
        }
        j
    }
}

impl SMO {
    pub fn run(
        &self,
        x: &Vec<Vec<f64>>,
        y: &Vec<f64>,
        kernel: &Box<dyn Kernel>,
    ) -> (Vec<f64>, f64) {
        let mut alphas = vec![0.0; x.len()];
        let mut b = 0.0;
        let mut passes = 0;
        let mut num_changed_alphas = 0;

        while passes < self.max_passes {
            num_changed_alphas = 0;
            for i in 0..x.len() {
                let e_i = SMO::calculate_error(&kernel, &x[i], y[i], b, &alphas, x, y);
                if (y[i] * e_i < -self.tol && alphas[i] < self.c)
                    || (y[i] * e_i > self.tol && alphas[i] > 0.0)
                {
                    let j = SMO::rand_j(x.len(), i);
                    let e_j = SMO::calculate_error(&kernel, &x[j], y[j], b, &alphas, x, y);
                    let alpha_i_old = alphas[i];
                    let alpha_j_old = alphas[j];
                    let (l, h) = if y[i] != y[j] {
                        (
                            SMO::compute_L(alpha_i_old, alpha_j_old, y[i], y[j], self.c),
                            SMO::compute_H(alpha_i_old, alpha_j_old, y[i], y[j], self.c),
                        )
                    } else {
                        (
                            SMO::compute_L(alpha_i_old, alpha_j_old, y[i], y[j], self.c),
                            SMO::compute_H(alpha_i_old, alpha_j_old, y[i], y[j], self.c),
                        )
                    };
                    if l == h {
                        continue;
                    }
                    let eta = SMO::calculate_eta(&kernel, &x[i], &x[j]);
                    if eta >= 0.0 {
                        continue;
                    }
                    let alpha_j_new = SMO::calculate_alpha_j(alpha_j_old, y[j], e_i, e_j, eta);
                    let alpha_j_new = SMO::clip_alpha_j(alpha_j_new, l, h);
                    if (alpha_j_new - alpha_j_old).abs() < 1e-5 {
                        continue;
                    }
                    let alpha_i_new =
                        SMO::calculate_alpha_i(alpha_i_old, y[i], y[j], alpha_j_old, alpha_j_new);
                    let (b1, b2) = SMO::calculate_bs(
                        b,
                        &x[i],
                        y[i],
                        &x[j],
                        y[j],
                        alpha_i_new,
                        alpha_j_new,
                        e_i,
                        e_j,
                        alpha_i_old,
                        alpha_j_old,
                        &kernel,
                    );
                    b = SMO::compute_b(b1, b2, alpha_i_new, alpha_j_new, self.c);
                    alphas[i] = alpha_i_new;
                    alphas[j] = alpha_j_new;
                    num_changed_alphas += 1;
                }
            }
            if num_changed_alphas == 0 {
                passes += 1;
            } else {
                passes = 0;
            }
        }
        (alphas, b)
    }
}
