use criterion::{black_box, criterion_group, criterion_main, Criterion};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::svm::svc::{SVCParameters, SVC as SmartcoreSVC};
use smartcore::svm::RBFKernel as SmartcoreRBFKernel;
use svm_burns::{RBFKernel as BurnsRBFKernel, SVC as BurnsSVC};

fn svm_burns_benchmark(c: &mut Criterion) {
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

    let rbf_kernel = BurnsRBFKernel::new(1.0);
    let mut parameters = svm_burns::Parameters::default();
    parameters.with_kernel(Box::new(rbf_kernel));
    let mut svc = BurnsSVC::new(parameters);

    c.bench_function("svm_burns", |b| {
        b.iter(|| {
            svm_burns::svm::SVM::fit(&mut svc, black_box(&x), black_box(&y));
        })
    });
}

fn smartcore_benchmark(c: &mut Criterion) {
    let x = DenseMatrix::from_2d_array(&[
        &[5.1, 3.5, 1.4, 0.2],
        &[4.9, 3.0, 1.4, 0.2],
        &[4.7, 3.2, 1.3, 0.2],
        &[4.6, 3.1, 1.5, 0.2],
        &[5.0, 3.6, 1.4, 0.2],
        &[5.4, 3.9, 1.7, 0.4],
        &[4.6, 3.4, 1.4, 0.3],
        &[5.0, 3.4, 1.5, 0.2],
        &[4.4, 2.9, 1.4, 0.2],
        &[4.9, 3.1, 1.5, 0.1],
        &[7.0, 3.2, 4.7, 1.4],
        &[6.4, 3.2, 4.5, 1.5],
        &[6.9, 3.1, 4.9, 1.5],
        &[5.5, 2.3, 4.0, 1.3],
        &[6.5, 2.8, 4.6, 1.5],
        &[5.7, 2.8, 4.5, 1.3],
        &[6.3, 3.3, 4.7, 1.6],
        &[4.9, 2.4, 3.3, 1.0],
        &[6.6, 2.9, 4.6, 1.3],
        &[5.2, 2.7, 3.9, 1.4],
    ]);

    let y: Vec<i32> = vec![
        -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ];

    let params =
        SVCParameters::default().with_kernel(SmartcoreRBFKernel::default().with_gamma(1.0));

    c.bench_function("smartcore_svc", |b| {
        b.iter(|| {
            let _ = SmartcoreSVC::fit(&x, &y, &params).unwrap();
        })
    });
}

criterion_group!(benches, svm_burns_benchmark, smartcore_benchmark);
criterion_main!(benches);
