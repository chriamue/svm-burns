pub trait SVM {
    fn fit(&mut self, x: &Vec<Vec<f64>>, y: &Vec<i32>);
    fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<i32>;
}
