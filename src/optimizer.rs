use crate::Kernel;

pub type AlphasB = (Vec<f64>, f64);

pub trait Optimizer {
    fn optimize(
        &mut self,
        x: &Vec<Vec<f64>>,
        y: &Vec<i32>,
        kernel: &Box<dyn Kernel>,
    ) -> (Vec<Vec<f64>>, Vec<f64>, f64);
}
