#![cfg(test)]

extern crate lda;
extern crate regex;
extern crate ordermap;
extern crate ndarray;
extern crate bidimap;

use ordermap::OrderMap;
use ndarray::{Array2, Axis};
use regex::Regex;
use bidimap::{HashBidiMap, BidiMap, MapLike};

fn parse_doc(text: &str, vocab: &HashBidiMap<&str, usize>) -> lda::Document {
    let mut ddict = OrderMap::new();
    let re_split = Regex::new(r"[^a-zA-Z]+").unwrap();
    let norm_text = Regex::new(r"[\.']").unwrap().replace_all(text, "").to_lowercase();
    for word in re_split.split(&norm_text[..]) {
        if let Some(idx) = vocab.as_map().get(&word) {
            *ddict.entry(*idx).or_insert(0_f32) += 1_f32;
        }
    }

    lda::Document { words: ddict }
}

fn get_topic<'a>(lambda: &Array2<f32>, idx: usize, vocab: &'a HashBidiMap<&str, usize>, n: usize) -> Vec<(&'a str, f32)> {
    let row = lambda.select(Axis(0), &[idx]);
    let sumk: f32 = row.into_iter().sum();
    let mut w: Vec<(usize, f32)> = row.into_iter().cloned().enumerate().collect();
    w.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    w.truncate(n);
    let mapped: Vec<_> = w.iter()
        .map(|&(idx, p)| (vocab.as_inv_map()[idx], p / sumk))
        .collect();

    mapped
}

#[test]
fn test_brocolli() {
    // This brocolli dataset is hypnotic.
    // https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html

    let docset = [
        "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother.",
        "My mother spends a lot of time driving my brother around to baseball practice.",
        "Some health experts suggest that driving may cause increased tension and blood pressure.",
        "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better.",
        "Health professionals say that brocolli is good for your health."        
    ];

    let words = "brocolli good eat brother mother spends time driving around \
        baseball practice health experts suggest increased tension blood pressure \
        feel perform well school seems better professionals say";

    let mut vocab = HashBidiMap::new();
    let words: Vec<_> = words.split(' ').zip((0..)).collect();
    vocab.extend(words);

    let d = docset.len();
    let w = vocab.len();
    let k = 2;
    let alpha = 1.0 / k as f32;
    let eta = 1.0 / k as f32;
    let tau0 = 1025.0;
    let kappa = 0.7;

    let mut olda = lda::OnlineLDA::new(w, k, d, alpha, eta, tau0, kappa);

    let mut perplexity = 0.0;
    for _it in 0..20 {
        for text in docset.iter() {
            let doc = parse_doc(text, &vocab);
            perplexity = olda.update_lambda_docs(&[doc]);
        }
    }

    assert_eq!(perplexity, 43.05503);

    assert_eq!(vec![
        ("health", 0.08495814),
        ("say", 0.050945763),
        ("professionals", 0.0508212),
        ("brocolli", 0.04943107),
        ("suggest", 0.048369262)],
        get_topic(olda.lambda(), 0, &vocab, 5));

    assert_eq!(vec![
        ("brother", 0.07146024),
        ("mother", 0.06900988),
        ("brocolli", 0.055741135),
        ("good", 0.05528625),
        ("eat", 0.054417454)],
        get_topic(olda.lambda(), 1, &vocab, 5));
}
