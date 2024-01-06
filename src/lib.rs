/// SVM implementation in Rust
///
/// # Sources
///
/// * Pattern recognition and machine learning, Bishop, 2006
/// * [Smartcore](https://github.com/smartcorelib/smartcore/blob/development/src/svm/svc.rs)
///
pub mod cache;
pub mod kernel;
pub mod optimizer;
pub mod parameters;
pub mod smo;
pub mod support_vector;
pub mod svc;
pub mod svm;

pub use kernel::Kernel;
pub use kernel::RBFKernel;
pub use parameters::Parameters;
pub use svc::SVC;

/// Used types
pub type X = Vec<Vec<f64>>;
pub type Y = Vec<i32>;
pub type W = Vec<f64>;
pub type B = f64;
