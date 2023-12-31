use criterion::{black_box, criterion_group, criterion_main, Criterion};
use smartcore::svm::RBFKernel as SmartcoreRBF;
use svm_burns::RBFKernel;

fn rbf_kernel_benchmark(c: &mut Criterion) {
    let svm_burns_rbf = RBFKernel::new(1.0);
    let smartcore_rbf = SmartcoreRBF::default().with_gamma(1.0);

    let x = vec![1.0, 2.0, 3.0];
    let y = vec![4.0, 5.0, 6.0];

    let mut group = c.benchmark_group("RBF Kernel Comparison");

    group.bench_function("svm_burns_rbf_kernel_compute", |b| {
        b.iter(|| {
            let result = svm_burns::Kernel::compute(&svm_burns_rbf, black_box(&x), black_box(&y));
            criterion::black_box(result);
        })
    });

    group.bench_function("smartcore_rbf_kernel_compute", |b| {
        b.iter(|| {
            let result =
                smartcore::svm::Kernel::apply(&smartcore_rbf, black_box(&x), black_box(&y))
                    .unwrap();
            criterion::black_box(result);
        })
    });

    group.finish();
}

criterion_group!(benches, rbf_kernel_benchmark);
criterion_main!(benches);
