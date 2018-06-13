use randomkit::dist::Gamma;
use randomkit::{Rng, Sample};
use statrs::function::gamma;

use ndarray::{Array1, Array2, Axis};

pub fn l1_dist_2d(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

pub fn exp_dirichlet_expectation_2d(array: &Array2<f64>) -> Array2<f64> {
    let mut d_exp = array.clone();

    let row_sums = array.sum_axis(Axis(1));
    let psi_row_sums = psi_1d(&row_sums);

    for (idx, mut row) in d_exp.genrows_mut().into_iter().enumerate() {
        row.mapv_inplace(|x| (psi(x) - psi_row_sums[idx]).exp());
    }

    d_exp
}

pub fn dirichlet_expectation_2d(array: &Array2<f64>) -> Array2<f64> {
    let mut d = array.clone();

    let row_sums = array.sum_axis(Axis(1));
    let psi_row_sums = psi_1d(&row_sums);

    for (idx, mut row) in d.genrows_mut().into_iter().enumerate() {
        row.mapv_inplace(|x| (psi(x) - psi_row_sums[idx]));
    }

    d
}

fn psi(x: f64) -> f64 {
    gamma::digamma(x)
}

fn psi_1d(vector: &Array1<f64>) -> Array1<f64> {
    vector.mapv(gamma::digamma)
}

pub fn gammaln(x: f64) -> f64 {
    gamma::ln_gamma(x)
}

pub fn gammaln_1d(vector: &Array1<f64>) -> Array1<f64> {
    vector.mapv(gamma::ln_gamma)
}

pub fn gammaln_2d(array: &Array2<f64>) -> Array2<f64> {
    array.mapv(gamma::ln_gamma)
}

pub fn random_gamma_array_2d(shape: f64, scale: f64, rows: usize, cols: usize, rng: &mut Rng) -> Array2<f64> {
    let dist = Gamma::new(shape, scale).unwrap();
    Array2::from_shape_fn((rows, cols), |_| dist.sample(rng))
}
