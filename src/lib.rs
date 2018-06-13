extern crate indexmap;
extern crate ndarray;
extern crate randomkit;
extern crate statrs;

use ndarray::{Array, Array1, Array2, Axis, Zip};
use randomkit::Rng;

use std::f64;
use indexmap::IndexMap;
use std::iter::FromIterator;

mod math;

#[derive(Debug)]
pub struct Document {
    pub words: IndexMap<usize, f64>,
}

impl Document {
    pub fn new() -> Self {
        Self {
            words: IndexMap::new(),
        }
    }
}

impl<'a> FromIterator<&'a usize> for Document {
    fn from_iter<I: IntoIterator<Item=&'a usize>>(iter: I) -> Self {
        let mut doc = Self::new();

        for i in iter {
            *doc.words.entry(*i).or_insert(0_f64) += 1_f64;
        }

        doc
    }
}

pub struct OnlineLDA {
    rng: Rng,
    // Vocabulary size
    w: usize,
    // Number of topics
    k: usize,
    // Total number of documents in the population
    d: usize,
    // Hyperparameter for prior on weight vectors theta
    alpha: f64,
    // Hyperparameter for prior on topics beta
    eta: f64,
    // A (positive) learning parameter that downweights early iterations
    tau0: f64,
    // Learning rate: exponential decay rate
    // should be between (0.5, 1.0] to guarantee asymptotic convergence.
    kappa: f64,
    // Weight in mini-batch
    rhot: f64,
    // Mini-batch iteration number
    updatect: f64,

    // Latent variables
    lambda: Array2<f64>,
    // E[log(beta)]
    elogbeta: Array2<f64>,
    // exp(E[log(beta)])
    expelogbeta: Array2<f64>,
}

pub struct OnlineLDABuilder {
    // Vocabulary size
    w: usize,
    // Number of topics
    k: usize,
    // Total number of documents in the population
    d: usize,
    // Hyperparameter for prior on weight vectors theta
    alpha: f64,
    // Hyperparameter for prior on topics beta
    eta: f64,
    // A (positive) learning parameter that downweights early iterations
    tau0: f64,
    // Learning rate: exponential decay rate
    // should be between (0.5, 1.0] to guarantee asymptotic convergence.
    kappa: f64,
}

impl OnlineLDABuilder {
    pub fn new(w: usize, d: usize, k: usize) -> Self {
        Self {
            w,
            d,
            k,
            alpha: 1.0 / k as f64,
            eta: 1.0 / k as f64,
            tau0: 1025.0,
            kappa: 0.7,
        }
    }

    pub fn alpha<'a>(&'a mut self, alpha: f64) -> &'a mut Self {
        self.alpha = alpha;
        self
    }

    pub fn eta<'a>(&'a mut self, eta: f64) -> &'a mut Self {
        self.eta = eta;
        self
    }

    pub fn tau0<'a>(&'a mut self, tau0: f64) -> &'a mut Self {
        self.tau0 = tau0;
        self
    }

    pub fn kappa<'a>(&'a mut self, kappa: f64) -> &'a mut Self {
        self.kappa = kappa;
        self
    }

    pub fn build(&mut self) -> OnlineLDA {
        let mut rng = Rng::from_seed(1);
        let lambda = math::random_gamma_array_2d(100.0, 0.01, self.k, self.w, &mut rng);
        let elogbeta = math::dirichlet_expectation_2d(&lambda);
        let expelogbeta = math::exp_dirichlet_expectation_2d(&lambda);

        OnlineLDA {
            rng,
            w: self.w,
            d: self.d,
            k: self.k,
            alpha: self.alpha,
            eta: self.eta,
            tau0: self.tau0,
            kappa: self.kappa,
            rhot: 0.0,
            updatect: 0.0,
            lambda,
            elogbeta,
            expelogbeta,
        }
    }
}

impl OnlineLDA {
    pub fn new(w: usize, k: usize, d: usize, alpha: f64, eta: f64, tau0: f64, kappa: f64) -> Self {
        let mut rng = Rng::from_seed(1);

        let rhot = 0.0;
        let updatect = 0.0;

        let lambda = math::random_gamma_array_2d(100.0, 0.01, k, w, &mut rng);
        let elogbeta = math::dirichlet_expectation_2d(&lambda);
        let expelogbeta = math::exp_dirichlet_expectation_2d(&lambda);

        Self { rng, w, k, d, alpha, eta, tau0, kappa, rhot, updatect, lambda, elogbeta, expelogbeta }
    }

    pub fn update_lambda_docs(&mut self, docs: &[Document]) -> f64 {
        self.rhot = (self.tau0 + self.updatect).powf(-self.kappa);
        let (gamma, sstats) = self.do_e_step_docs(docs);
        let bound = self.approx_bound_docs(docs, &gamma);

        self.lambda = self.lambda.clone() * (1.0 - self.rhot) +
            self.rhot * (self.eta + (self.d as f64) * sstats / (docs.len() as f64));
        self.elogbeta = math::dirichlet_expectation_2d(&self.lambda);
        self.expelogbeta = math::exp_dirichlet_expectation_2d(&self.lambda);

        self.updatect += 1.0;

        let words: Vec<f64> = docs
            .into_iter()
            .map(|doc| doc.words.values().sum())
            .collect();
        let words_count: f64 = words.iter().sum();

        let perwordbound: f64 = bound * (docs.len() as f64) / ((self.d as f64) * words_count);
        let perplexity = (-perwordbound).exp();

        perplexity
    }

    pub fn approx_bound_docs(&mut self, docs: &[Document], gamma: &Array2<f64>) -> f64 {
        let batch_d = docs.len();

        let mut score = 0_f64;
        let elogtheta = math::dirichlet_expectation_2d(&gamma);

        // E[log p(docs | theta, beta)]
        for d in 0..batch_d {
            let ids: Vec<usize> = docs[d].words.keys().cloned().collect();
            let w = ids.len();
            let cts: Vec<f64> = ids.iter().map(|k| docs[d].words[k]).collect();
            let cts_1d: Array1<f64> = Array::from_vec(cts.clone());
            let mut phinorm = Array1::<f64>::zeros(w);
            for i in 0..w {
                let mut temp = elogtheta.select(Axis(0), &[d]) + self.elogbeta.select(Axis(1), &[ids[i]]).t();
                let tmax = temp.fold(f64::MIN, |a, b| a.max(*b));
                temp.mapv_inplace(|x| (x - tmax).exp());
                phinorm[i] = temp.scalar_sum().ln() + tmax;
            }

            score += (cts_1d * phinorm).scalar_sum();
        }

        // E[log p(theta | alpha) - log q(theta | gamma)]
        score += ((self.alpha - gamma.clone())*elogtheta).scalar_sum();
        score += (math::gammaln_2d(&gamma) - math::gammaln(self.alpha)).scalar_sum();
        score += (math::gammaln(self.alpha * (self.k as f64)) - math::gammaln_1d(&gamma.sum_axis(Axis(1)))).scalar_sum();

        // Compensate for the subsampling of the population of documents
        score = score * (self.d as f64) / (docs.len() as f64);

        // E[log p(beta | eta) - log q (beta | lambda)]
        score += ((self.eta - &self.lambda) * &self.elogbeta).scalar_sum();
        score += (math::gammaln_2d(&self.lambda) - math::gammaln(self.eta)).scalar_sum();
        score += (math::gammaln(self.eta * (self.w as f64)) - math::gammaln_1d(&self.lambda.sum_axis(Axis(1)))).scalar_sum();

        score
    }

    fn do_e_step_docs(&mut self, docs: &[Document]) -> (Array2<f64>, Array2<f64>) {
        let batch_d = docs.len();

        // Initialize the variational distribution q(theta|gamma) for
        // the mini-batch
        let mut gamma = math::random_gamma_array_2d(100.0, 0.01, batch_d, self.k, &mut self.rng);
        let expelogtheta = math::exp_dirichlet_expectation_2d(&gamma);

        let mut sstats = Array2::<f64>::zeros(self.lambda.dim());

        // Now, for each document d update that document's gamma and phi
        for d in 0..batch_d {
            let ids: Vec<usize> = docs[d].words.keys().cloned().collect();
            let w = ids.len();
            let cts: Vec<f64> = ids.iter().map(|k| docs[d].words[k]).collect();
            let cts_2d: Array2<f64> = Array::from_vec(cts).into_shape((1, w)).unwrap();

            let mut gammad = gamma.select(Axis(0), &[d]);
            let mut expelogthetad = expelogtheta.select(Axis(0), &[d]);
            let expelogbetad = self.expelogbeta.select(Axis(1), &ids[..]);
            // The optimal phi_{dwk} is proportional to 
            // expElogthetad_k * expElogbetad_w. phinorm is the normalizer.
            let mut phinorm = expelogthetad.dot(&expelogbetad) + f64::EPSILON;

            let meanchangethresh = 0.001;
            // Iterate between gamma and phi until convergence
            for _it in 0..100 {
                let lastgamma = gammad.clone();
                // Substituting the value of the optimal phi back into
                // the update for gamma gives this update. Cf. Lee&Seung 2001.
                gammad = self.alpha + expelogthetad.clone() *
                    (cts_2d.clone() / phinorm).dot(&expelogbetad.t());

                expelogthetad = math::exp_dirichlet_expectation_2d(&gammad);
                phinorm = expelogthetad.dot(&expelogbetad) + f64::EPSILON;

                if (math::l1_dist_2d(&lastgamma, &gammad) / (self.k as f64)) < meanchangethresh {
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

    pub fn get_topic_top_n<'a>(&self, k: usize, n: usize) -> Vec<(usize, f64)> {
        let row = self.lambda.select(Axis(0), &[k]);
        let sumk: f64 = row.into_iter().sum();
        let mut w: Vec<(usize, f64)> = row.into_iter().cloned().enumerate().collect();
        w.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        w.truncate(n);
        let mapped: Vec<_> = w.iter()
            .map(|&(idx, p)| (idx, p / sumk))
            .collect();

        mapped
    }
}
