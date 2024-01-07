// source: https://chubakbidpaa.com/svm/2020/12/27/smo-algorithm-simplifed-copy.html
// source: https://github.com/smartcorelib/smartcore/blob/development/src/svm/svc.rs

use std::collections::HashSet;

use rand::{seq::SliceRandom, SeedableRng};
#[cfg(feature = "parallel")]
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{cache::Cache, optimizer::Optimizer, support_vector::SupportVector, Kernel};

#[derive(Debug)]
pub struct SMO {
    /// regularization parameter
    c: f64,
    /// numerical tolerance
    tol: f64,
    /// maximum number of iterations over Larange multipliers without changing
    epochs: usize,

    seed: Option<usize>,

    cache: Cache,

    sv: Vec<SupportVector>,

    min_grad: f64,
    max_grad: f64,
    min_sv: usize,
    max_sv: usize,
    recalculate_min_max: bool,
}

unsafe impl Sync for SMO {}
unsafe impl Send for SMO {}

impl SMO {
    pub fn new(c: f64, tol: f64, epochs: usize) -> Self {
        SMO {
            c,
            tol,
            epochs,
            cache: Cache::new(),
            seed: None,
            sv: Vec::new(),
            min_grad: f64::INFINITY,
            max_grad: f64::NEG_INFINITY,
            min_sv: 0,
            max_sv: 0,
            recalculate_min_max: true,
        }
    }
}

impl SMO {
    pub fn with_c(&mut self, c: f64) -> &mut Self {
        self.c = c;
        self
    }

    pub fn with_tol(&mut self, tol: f64) -> &mut Self {
        self.tol = tol;
        self
    }

    pub fn with_epochs(&mut self, epochs: usize) -> &mut Self {
        self.epochs = epochs;
        self
    }

    pub fn with_seed(&mut self, seed: Option<usize>) -> &mut Self {
        self.seed = seed;
        self
    }
}

impl Default for SMO {
    fn default() -> Self {
        SMO::new(1.0, 1e-3, 5)
    }
}

impl SMO {
    pub fn initialize(&mut self, x: &Vec<Vec<f64>>, y: &Vec<i32>, kernel: &Box<dyn Kernel>) {
        let n = x.len();
        let few = 5;
        let mut cp = 0; // count of positive samples
        let mut cn = 0; // count of positive and negative samples

        let mut x_new: Vec<f64> = Vec::with_capacity(n);

        for i in Self::permutate(n, self.seed) {
            x_new.clear();
            x_new.extend(x[i].iter().take(n).copied());

            if y[i] == 1 && cp < few && self.process(i, &x_new, y[i], kernel) {
                cp += 1;
            } else if y[i] == -1 && cn < few && self.process(i, &x_new, y[i], kernel) {
                cn += 1;
            }

            if cp >= few && cn >= few {
                break;
            }
        }
    }

    fn process(&mut self, i: usize, x: &Vec<f64>, y: i32, kernel: &Box<dyn Kernel>) -> bool {
        for j in 0..self.sv.len() {
            if self.sv[j].index == i {
                return true;
            }
        }

        let mut g: f64 = y.into();
        let mut cache_values: Vec<((usize, usize), f64)> = Vec::new();

        for v in self.sv.iter() {
            let xi = &v.x;
            let xj = x;
            let k = kernel.compute(xi, xj);
            cache_values.push(((i, v.index), k));
            g -= v.alpha * k;
        }

        self.find_min_max_gradient();

        if self.min_grad < self.max_grad
            && ((y > 0 && g < self.min_grad) || (y < 0 && g > self.max_grad))
        {
            return false;
        }

        for v in cache_values {
            self.cache.insert(v.0, v.1);
        }

        let k_v = kernel.compute(x, x);

        self.sv.insert(
            0,
            SupportVector::new(i, x.to_vec(), y.into(), g, self.c, k_v),
        );

        if y > 0 {
            self.smo(None, Some(0), 0.0, kernel);
        } else {
            self.smo(Some(0), None, 0.0, kernel);
        }
        true
    }

    fn reprocess(&mut self, kernel: &Box<dyn Kernel>) -> bool {
        let status = self.smo(None, None, self.tol, kernel);
        self.clean();
        status
    }

    fn finish(&mut self, kernel: &Box<dyn Kernel>) {
        let mut max_iter = self.sv.len();

        while self.smo(None, None, self.tol, kernel) && max_iter > 0 {
            max_iter -= 1;
        }

        self.clean();
    }

    fn find_min_max_gradient(&mut self) {
        if !self.recalculate_min_max {
            return;
        }

        for i in 0..self.sv.len() {
            let v = &self.sv[i];
            let grad = v.grad;
            let alpha = v.alpha;
            if grad < self.min_grad && alpha > v.cmin {
                self.min_grad = grad;
                self.min_sv = i;
            }
            if grad > self.max_grad && alpha < v.cmax {
                self.max_grad = grad;
                self.max_sv = i;
            }
        }

        self.recalculate_min_max = false;
    }

    fn clean(&mut self) {
        self.find_min_max_gradient();
        let max_grad = self.max_grad;
        let min_grad = self.min_grad;

        let mut idxs_to_drop = HashSet::new();

        self.sv.retain(|v| {
            let alpha = v.alpha;
            let cmin = v.cmin;
            let cmax = v.cmax;
            let grad = v.grad;
            let idx = v.index;
            if alpha == 0.0
                && ((grad >= max_grad && 0.0 >= cmax) || (grad <= min_grad && 0.0 <= cmin))
            {
                idxs_to_drop.insert(idx);
                return false;
            }
            true
        });
        self.cache.drop_all(idxs_to_drop);
        self.recalculate_min_max = true;
    }

    /// permute the indices of the support vectors
    pub fn permutate(n: usize, seed: Option<usize>) -> Vec<usize> {
        let mut rng = match seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed as u64),
            None => rand::rngs::StdRng::from_entropy(),
        };
        let mut perm: Vec<usize> = (0..n).collect();
        perm.shuffle(&mut rng);
        perm
    }

    fn select_pair(
        &mut self,
        idx_1: Option<usize>,
        idx_2: Option<usize>,
        kernel: &Box<dyn Kernel>,
    ) -> Option<(usize, usize, f64)> {
        match (idx_1, idx_2) {
            (None, None) => {
                if self.max_grad > -self.min_grad {
                    self.select_pair(None, Some(self.max_sv), kernel)
                } else {
                    self.select_pair(Some(self.min_sv), None, kernel)
                }
            }
            (Some(idx_1), None) => {
                let sv1 = &self.sv[idx_1];
                let mut idx_2 = None;
                let mut k_v_12: Option<f64> = None;
                let km = sv1.k;
                let gm = sv1.grad;
                let mut best = 0f64;
                let xi = &sv1.x;
                for i in 0..self.sv.len() {
                    let v = &self.sv[i];
                    let xj = &v.x;
                    let z = v.grad - gm;
                    let k = self
                        .cache
                        .get_or_insert((sv1.index, v.index), kernel.compute(&xi, &xj));
                    let mut curv = km + v.k - 2.0 * k;
                    if curv <= 0.0 {
                        curv = 1e-12; // tau
                    }
                    let mu = z / curv;
                    if (mu > 0.0 && v.alpha < v.cmax) || (mu < 0.0 && v.alpha > v.cmin) {
                        let gain = z * mu;
                        if gain > best {
                            best = gain;
                            idx_2 = Some(i);
                            k_v_12 = Some(*k);
                        }
                    }
                }

                let xi = &self.sv[idx_1].x;
                idx_2.map(|idx_2| {
                    (
                        idx_1,
                        idx_2,
                        k_v_12.unwrap_or_else(|| kernel.compute(&xi, &self.sv[idx_2].x)),
                    )
                })
            }
            (None, Some(idx_2)) => {
                let mut idx_1 = None;
                let sv2 = &self.sv[idx_2];
                let mut k_v_12: Option<f64> = None;
                let km = sv2.k;
                let gm = sv2.grad;
                let mut best = 0f64;

                let xi = &sv2.x;
                for i in 0..self.sv.len() {
                    let v = &self.sv[i];
                    let xj = &v.x;
                    let z = gm - v.grad;
                    let k = self
                        .cache
                        .get_or_insert((sv2.index, v.index), kernel.compute(&xi, &xj));
                    let mut curv = km + v.k - 2.0 * k;
                    if curv <= 0.0 {
                        curv = 1e-12; // tau
                    }
                    let mu = z / curv;
                    if (mu > 0.0 && v.alpha > v.cmin) || (mu < 0.0 && v.alpha < v.cmax) {
                        let gain = z * mu;
                        if gain > best {
                            best = gain;
                            idx_1 = Some(i);
                            k_v_12 = Some(*k);
                        }
                    }
                }

                let xj = &self.sv[idx_2].x;
                idx_1.map(|idx_1| {
                    (
                        idx_1,
                        idx_2,
                        k_v_12.unwrap_or_else(|| kernel.compute(&self.sv[idx_1].x, &xj)),
                    )
                })
            }
            (Some(idx_1), Some(idx_2)) => Some((
                idx_1,
                idx_2,
                kernel.compute(&self.sv[idx_1].x, &self.sv[idx_2].x),
            )),
        }
    }

    fn smo(
        &mut self,
        idx_1: Option<usize>,
        idx_2: Option<usize>,
        tol: f64,
        kernel: &Box<dyn Kernel>,
    ) -> bool {
        match self.select_pair(idx_1, idx_2, kernel) {
            Some((idx_1, idx_2, k_v_12)) => {
                let mut curv = self.sv[idx_1].k + self.sv[idx_2].k - 2.0 * k_v_12;
                if curv >= 0.0 {
                    curv = 1e-12; // tau
                }
                let mut step = (self.sv[idx_2].grad - self.sv[idx_1].grad) / curv;
                if step >= 0.0 {
                    let mut ostep = self.sv[idx_1].alpha - self.sv[idx_1].cmin;
                    if ostep < step {
                        step = ostep;
                    }
                    ostep = self.sv[idx_2].cmax - self.sv[idx_2].alpha;
                    if ostep < step {
                        step = ostep;
                    }
                } else {
                    let mut ostep = self.sv[idx_2].cmin - self.sv[idx_2].alpha;
                    if ostep > step {
                        step = ostep;
                    }
                    ostep = self.sv[idx_1].alpha - self.sv[idx_1].cmax;
                    if ostep > step {
                        step = ostep;
                    }
                }
                self.update(idx_1, idx_2, step, kernel);
                self.max_grad - self.min_grad > tol
            }
            None => false,
        }
    }

    fn update(&mut self, v1: usize, v2: usize, step: f64, kernel: &Box<dyn Kernel>) {
        self.sv[v1].alpha -= step;
        self.sv[v2].alpha += step;

        let xi_v1 = self.sv[v1].x.clone();
        let xi_v2 = self.sv[v2].x.clone();
        let sv_v1_index = self.sv[v1].index;
        let sv_v2_index = self.sv[v2].index;

        // Precompute k1 and k2 values outside the loop
        let mut k1_values = Vec::new();
        let mut k2_values = Vec::new();
        for i in 0..self.sv.len() {
            let xj = &self.sv[i].x;
            k2_values.push(
                *self
                    .cache
                    .get_or_insert((sv_v2_index, self.sv[i].index), kernel.compute(&xi_v2, &xj)),
            );
            k1_values.push(
                *self
                    .cache
                    .get_or_insert((sv_v1_index, self.sv[i].index), kernel.compute(&xi_v1, &xj)),
            );
        }

        for i in 0..self.sv.len() {
            let k2 = k2_values[i];
            let k1 = k1_values[i];
            self.sv[i].grad -= step * (k2 - k1);
        }

        self.recalculate_min_max = true;
        self.find_min_max_gradient();
    }
}

impl Optimizer for SMO {
    fn optimize(
        &mut self,
        x: &Vec<Vec<f64>>,
        y: &Vec<i32>,
        kernel: &Box<dyn Kernel>,
    ) -> (Vec<Vec<f64>>, Vec<f64>, f64) {
        let n = x.len();

        self.cache = Cache::new();

        self.initialize(x, y, kernel);

        let good_enough = 1000.0;

        let mut x_new = Vec::with_capacity(n);
        for _ in 0..self.epochs {
            for i in Self::permutate(n, self.seed) {
                x_new.clear();
                x_new.extend(x[i].iter().take(n).copied());
                self.process(i, &x_new, y[i], kernel);
                loop {
                    self.reprocess(kernel);
                    self.find_min_max_gradient();
                    if self.max_grad - self.min_grad < good_enough {
                        break;
                    }
                }
            }
        }

        self.finish(kernel);
        let mut support_vectors = Vec::new();
        let mut w = Vec::new();
        let b = (self.min_grad + self.max_grad) / 2.0;

        for v in self.sv.iter() {
            support_vectors.push(v.x.clone());
            w.push(v.alpha);
        }

        (support_vectors, w, b)
    }
}
