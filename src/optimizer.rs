use crate::Kernel;

pub type AlphasB = (Vec<f64>, f64);

pub trait Optimizer {
    fn optimize(&self, x: &Vec<Vec<f64>>, y: &Vec<f64>, kernel: &Box<dyn Kernel>) -> AlphasB;
}
