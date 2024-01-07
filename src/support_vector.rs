use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct SupportVector {
    pub index: usize,
    pub x: Vec<f64>,
    pub alpha: f64,
    pub grad: f64,
    pub cmin: f64,
    pub cmax: f64,
    pub k: f64,
}

impl SupportVector {
    pub fn new(index: usize, x: Vec<f64>, y: f64, grad: f64, c: f64, k: f64) -> SupportVector {
        let (cmin, cmax) = if y > 0.0 { (0.0, c) } else { (-c, 0.0) };
        SupportVector {
            index,
            x,
            grad,
            k,
            alpha: 0.0,
            cmin,
            cmax,
        }
    }
}

/// Convert a SupportVector into a vector of f64
impl From<SupportVector> for Vec<f64> {
    fn from(sv: SupportVector) -> Self {
        sv.x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_support_vector() {
        let x = vec![1.0, 2.0, 3.0];
        let y = 1.0;
        let grad = 0.1;
        let c = 1.0;
        let k = 0.5;

        let sv = SupportVector::new(0, x.clone(), y, grad, c, k);

        assert_eq!(x, sv.x);
        assert_eq!(grad, sv.grad);
        assert_eq!(k, sv.k);
        assert_eq!(0.0, sv.cmin);
        assert_eq!(1.0, sv.cmax);
    }
}
