extern crate wikipedia;

extern crate itertools;
extern crate regex;
extern crate ordermap;
extern crate ndarray;

extern crate lda;

use ndarray::Array2;

use regex::Regex;

use std::path::Path;
use std::fs;
use std::fs::File;
use std::io::Read;
use std::collections::hash_map::HashMap;
use ordermap::OrderMap;

fn parse_doc(text: &str, vocab: &HashMap<&str, usize>) -> lda::Document {
    let mut ddict = OrderMap::new();
    let re_split = Regex::new(r"[^a-zA-Z]+").unwrap();
    let norm_text = Regex::new(r"[\.']").unwrap().replace_all(text, "").to_lowercase();
    for word in re_split.split(&norm_text[..]) {
        if vocab.contains_key(word) {
            let idx = vocab.get(word).unwrap();
            *ddict.entry(*idx).or_insert(0_f32) += 1_f32;
        } else {
            // oow, skip
        }
    }

    lda::Document { words: ddict }
}

fn print_topics(lambda: &Array2<f32>, vocab: &HashMap<usize, &str>, n: usize) -> () {
    let mut iteration = 0;
    for row in lambda.genrows() {
        let lambdak = row.clone().to_vec();
        let sumk: f32 = lambdak.iter().sum();
        let mut w: Vec<(usize, f32)> = lambdak.into_iter().enumerate().collect();
        w.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        w.truncate(n);
        let mapped: Vec<_> = w.iter()
            .map(|&(idx, p)| (vocab.get(&idx).unwrap(), p / sumk))
            .collect();

        iteration += 1;
        println!("topic {}:", iteration);
        for (k, p) in mapped.into_iter() {
            println!("  {0: <20}  \t---\t  {1:.4}", k, p);
        }
        println!("");
    }
}

fn read_file(path: &Path) -> String {
    let mut f = File::open(path).unwrap();
    let mut text = String::new();
    f.read_to_string(&mut text).unwrap();
    text
}

fn main() {
    // vocabulary
    let text = read_file(Path::new("./examples/dictnostops.txt"));
    let vocab: HashMap<&str, usize> = text.split('\n').zip((0..)).collect();
    let vocab2: HashMap<usize, &str> = text.split('\n').enumerate().collect();

    // settings
    let d = 1000;
    let w = vocab.len();
    let k = 10; // The number of topics
    let alpha = 1.0 / k as f32;
    let eta = 1.0 / k as f32;
    let tau0 = 1025.0;
    let kappa = 0.7;

    // init
    let mut olda = lda::OnlineLDA::new(w, k, d, alpha, eta, tau0, kappa);

    // feed data
    let dir = fs::read_dir("./examples/data").unwrap();
    for (it, entry) in dir.take(d).enumerate() {
        let path = entry.unwrap().path();
        let text = read_file(&path);
        let doc = parse_doc(text.as_str(), &vocab);

        let perplexity = olda.update_lambda_docs(&[doc]);
        println!("{}: held-out perplexity estimate = {}", it, perplexity);
    }

    print_topics(olda.lambda(), &vocab2, 10);
}
