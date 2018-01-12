extern crate itertools;
extern crate rand;
extern crate ordermap;
extern crate ndarray;

use ndarray::{Array, Array1, Array2, Axis, Zip};
use rand::{StdRng, SeedableRng};

use std::f32;
use ordermap::OrderMap;

mod math;

#[derive(Debug)]
pub struct Document {
    pub words: OrderMap<usize, f32>,
}

pub struct OnlineLDA {
    rng: StdRng,
    // Vocabulary size
    w: usize,
    // Number of topics
    k: usize,
    // Total number of documents in the population
    d: usize,
    // Hyperparameter for prior on weight vectors theta
    alpha: f32,
    // Hyperparameter for prior on topics beta
    eta: f32,
    // A (positive) learning parameter that downweights early iterations
    tau0: f32,
    // Learning rate: exponential decay rate
    kappa: f32,
    // Weight in mini-batch
    rhot: f32,
    // Mini-batch iteration number
    updatect: f32,

    // Latent variables
    lambda: Array2<f32>,
    // E[log(beta)]
    elogbeta: Array2<f32>,
    // exp(E[log(beta)])
    expelogbeta: Array2<f32>,
}

impl OnlineLDA {
    pub fn new(w: usize, k: usize, d: usize, alpha: f32, eta: f32, tau0: f32, kappa: f32) -> Self {
        let seed: &[_] = &[0, 0, 0, 1];
        let mut rng = SeedableRng::from_seed(seed);

        let rhot = 0.0;
        let updatect = 0.0;

        let lambda = math::random_gamma_array_2d(100.0, 0.01, k, w, &mut rng);
        let elogbeta = math::dirichlet_expectation_2d(&lambda);
        let expelogbeta = math::exp_dirichlet_expectation_2d(&lambda);

        Self { rng, w, k, d, alpha, eta, tau0, kappa, rhot, updatect, lambda, elogbeta, expelogbeta }
    }

    pub fn update_lambda_docs(&mut self, docs: &[Document]) -> f32 {
        self.rhot = (self.tau0 + self.updatect).powf(-self.kappa);
        let (gamma, sstats) = self.do_e_step_docs(docs);
        let bound = self.approx_bound_docs(docs, &gamma);

        self.lambda = self.lambda.clone() * (1.0 - self.rhot) +
            self.rhot * (self.eta + (self.d as f32) * sstats / (docs.len() as f32));
        self.elogbeta = math::dirichlet_expectation_2d(&self.lambda);
        self.expelogbeta = math::exp_dirichlet_expectation_2d(&self.lambda);

        self.updatect += 1.0;

        let words: Vec<f32> = docs
            .into_iter()
            .map(|doc| doc.words.values().sum())
            .collect();
        let words_count: f32 = words.iter().sum();

        let perwordbound: f32 = bound * (docs.len() as f32) / ((self.d as f32) * words_count);
        let perplexity = (-perwordbound).exp();

        perplexity
    }

    pub fn approx_bound_docs(&mut self, docs: &[Document], gamma: &Array2<f32>) -> f32 {
        let batch_d = docs.len();

        let mut score = 0_f32;
        let elogtheta = math::dirichlet_expectation_2d(&gamma);

        // E[log p(docs | theta, beta)]
        for d in 0..batch_d {
            let ids: Vec<usize> = docs[d].words.keys().cloned().collect();
            let w = ids.len();
            let cts: Vec<f32> = ids.iter().map(|k| docs[d].words[k]).collect();
            let cts_1d: Array1<f32> = Array::from_vec(cts.clone());
            let mut phinorm = Array1::<f32>::zeros(w);
            for i in 0..w {
                let mut temp = elogtheta.select(Axis(0), &[d]) + self.elogbeta.select(Axis(1), &[ids[i]]).t();
                let tmax = temp.fold(f32::MIN, |a, b| a.max(*b));
                temp.mapv_inplace(|x| (x - tmax).exp());
                phinorm[i] = temp.scalar_sum().ln() + tmax;
            }

            score += (cts_1d * phinorm).scalar_sum();
        }

        // E[log p(theta | alpha) - log q(theta | gamma)]
        score += ((self.alpha - gamma.clone())*elogtheta).scalar_sum();
        score += (math::gammaln_2d(&gamma) - math::gammaln(self.alpha)).scalar_sum();
        score += (math::gammaln(self.alpha * (self.k as f32)) - math::gammaln_1d(&gamma.sum_axis(Axis(1)))).scalar_sum();

        // Compensate for the subsampling of the population of documents
        score = score * (self.d as f32) / (docs.len() as f32);

        // E[log p(beta | eta) - log q (beta | lambda)]
        score += ((self.eta - &self.lambda) * &self.elogbeta).scalar_sum();
        score += (math::gammaln_2d(&self.lambda) - math::gammaln(self.eta)).scalar_sum();
        score += (math::gammaln(self.eta * (self.w as f32)) - math::gammaln_1d(&self.lambda.sum_axis(Axis(1)))).scalar_sum();

        score
    }

    pub fn do_e_step_docs(&mut self, docs: &[Document]) -> (Array2<f32>, Array2<f32>) {
        let batch_d = docs.len();

        // Initialize the variational distribution q(theta|gamma) for
        // the mini-batch
        let mut gamma = math::random_gamma_array_2d(100.0, 0.01, batch_d, self.k, &mut self.rng);
        let expelogtheta = math::exp_dirichlet_expectation_2d(&gamma);

        let mut sstats = Array2::<f32>::zeros(self.lambda.dim());

        // Now, for each document d update that document's gamma and phi
        for d in 0..batch_d {
            let ids: Vec<usize> = docs[d].words.keys().cloned().collect();
            let w = ids.len();
            let cts: Vec<f32> = ids.iter().map(|k| docs[d].words[k]).collect();
            let cts_2d: Array2<f32> = Array::from_vec(cts).into_shape((1, w)).unwrap();

            let mut gammad = gamma.select(Axis(0), &[d]);
            let mut expelogthetad = expelogtheta.select(Axis(0), &[d]);
            let expelogbetad = self.expelogbeta.select(Axis(1), &ids[..]);
            // The optimal phi_{dwk} is proportional to 
            // expElogthetad_k * expElogbetad_w. phinorm is the normalizer.
            let mut phinorm = expelogthetad.dot(&expelogbetad) + f32::EPSILON;

            let meanchangethresh = 0.001;
            // Iterate between gamma and phi until convergence
            for _it in 0..100 {
                let lastgamma = gammad.clone();
                // Substituting the value of the optimal phi back into
                // the update for gamma gives this update. Cf. Lee&Seung 2001.
                gammad = self.alpha + expelogthetad.clone() *
                    (cts_2d.clone() / phinorm).dot(&expelogbetad.t());

                expelogthetad = math::exp_dirichlet_expectation_2d(&gammad);
                phinorm = expelogthetad.dot(&expelogbetad) + f32::EPSILON;

                if (math::l1_dist_2d(&lastgamma, &gammad) / (self.k as f32)) < meanchangethresh {
                    break;
                }
            }

            gamma.subview_mut(Axis(0), d).assign(&gammad.row(0));

            let sstatsd = expelogthetad.t().dot(&(cts_2d.clone() / phinorm));
            Zip::from(sstats.genrows_mut())
                .and(sstatsd.genrows())
                .apply(|mut a_row, b_row| {
                    for (idx, num) in ids.iter().enumerate() {
                        a_row[*num] += b_row[idx];
                    }
                });
        }

        sstats = sstats * &self.expelogbeta;

        (gamma, sstats)
    }

    pub fn lambda(&self) -> &Array2<f32> {
        &self.lambda
    }
}
