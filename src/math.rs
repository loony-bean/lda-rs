extern crate rand;
extern crate ndarray;
extern crate ndarray_rand;
extern crate fastapprox;

use rand::StdRng;
use rand::distributions::Gamma;

use ndarray::{Array, Array1, Array2, Axis};
use self::ndarray_rand::{RandomExt, F32};

use self::fastapprox::fast;

pub fn l1_dist_2d(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

pub fn exp_dirichlet_expectation_2d(array: &Array2<f32>) -> Array2<f32> {
    let mut d_exp = array.clone();

    let row_sums = array.sum_axis(Axis(1));
    let psi_row_sums = psi_1d(&row_sums);

    for (idx, mut row) in d_exp.genrows_mut().into_iter().enumerate() {
        row.mapv_inplace(|x| (psi(x) - psi_row_sums[idx]).exp());
    }

    d_exp
}

pub fn dirichlet_expectation_2d(array: &Array2<f32>) -> Array2<f32> {
    let mut d = array.clone();

    let row_sums = array.sum_axis(Axis(1));
    let psi_row_sums = psi_1d(&row_sums);

    for (idx, mut row) in d.genrows_mut().into_iter().enumerate() {
        row.mapv_inplace(|x| (psi(x) - psi_row_sums[idx]));
    }

    d
}

fn psi(x: f32) -> f32 {
    fast::digamma(x)
}

fn psi_1d(vector: &Array1<f32>) -> Array1<f32> {
    vector.mapv(fast::digamma)
}

pub fn gammaln(x: f32) -> f32 {
    fast::ln_gamma(x)
}

pub fn gammaln_1d(vector: &Array1<f32>) -> Array1<f32> {
    vector.mapv(fast::ln_gamma)
}

pub fn gammaln_2d(array: &Array2<f32>) -> Array2<f32> {
    array.mapv(fast::ln_gamma)
}

pub fn random_gamma_array_2d(shape: f64, scale: f64, rows: usize, cols: usize, rng: &mut StdRng) -> Array2<f32> {
    Array::random_using((rows, cols), F32(Gamma::new(shape, scale)), rng)
}
