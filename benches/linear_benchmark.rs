use criterion::{black_box, criterion_group, criterion_main, Criterion};
use smartcore::svm::LinearKernel as SmartcoreLinear;
use svm_burns::kernel::LinearKernel;

fn linear_kernel_benchmark(c: &mut Criterion) {
    let svm_burns_kernel = LinearKernel::default();
    let smartcore_kernel = SmartcoreLinear::default();

    let x = vec![1.0, 2.0, 3.0];
    let y = vec![4.0, 5.0, 6.0];

    let mut group = c.benchmark_group("Linear Kernel Comparison");

    group.bench_function("svm_burns_linear_kernel_compute", |b| {
        b.iter(|| {
            let result =
                svm_burns::Kernel::compute(&svm_burns_kernel, black_box(&x), black_box(&y));
            criterion::black_box(result);
        })
    });

    group.bench_function("smartcore_linear_kernel_compute", |b| {
        b.iter(|| {
            let result =
                smartcore::svm::Kernel::apply(&smartcore_kernel, black_box(&x), black_box(&y))
                    .unwrap();
            criterion::black_box(result);
        })
    });

    group.finish();
}

criterion_group!(benches, linear_kernel_benchmark);
criterion_main!(benches);
