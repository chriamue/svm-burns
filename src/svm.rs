pub trait SVM {
    fn fit(&mut self, x: &Vec<Vec<f64>>, y: &Vec<f64>);
    fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<f64>;
}
