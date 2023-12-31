# svm-burns
Integrating the efficiency of the Burn machine learning framework with a Support Vector Machine (SVM) implementation for advanced binary classification tasks.

## Project Status

This project is currently in development.

## Quickstart

### Build

```bash
cargo build
```

### Test

```bash
cargo test
```

### Benchmark

```bash
cargo bench
```

Usage of rayon gives a good speedup on the benchmark.

```bash
cargo bench --features parallel

Running benches/svm_benchmark.rs (target/release/deps/svm_benchmark-c30a44e1ad19bf51)
svm_burns               time:   [19.193 µs 19.267 µs 19.346 µs]
                        change: [-87.917% -87.836% -87.748%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 9 outliers among 100 measurements (9.00%)
  3 (3.00%) low mild
  5 (5.00%) high mild
  1 (1.00%) high severe

smartcore_svc           time:   [259.40 µs 259.50 µs 259.61 µs]
                        change: [+0.7936% +0.8892% +0.9764%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 3 outliers among 100 measurements (3.00%)
  2 (2.00%) high mild
  1 (1.00%) high severe
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Sources

- [Smartcore](https://github.com/smartcorelib/smartcore/blob/development/src/svm/mod.rs)
