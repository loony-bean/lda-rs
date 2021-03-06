#![cfg(test)]

extern crate lda;

use std::collections::HashMap;

fn parse_doc(text: &str, vocab: &HashMap<&str, usize>) -> lda::Document {
    let words = text
        .split(|c: char| !c.is_alphabetic())
        .map(|s| s.to_lowercase())
        .filter_map(|s| vocab.get(&*s));

    words.collect()
}

fn parse_topic<'a>(topic: &Vec<(usize, f32)>, vocab: &'a HashMap<usize, &str>) -> Vec<(&'a str, f32)> {
    topic.iter()
        .map(|&(idx, p)| (vocab[&idx], p))
        .collect()
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

    let v: HashMap<&str, usize> = words
        .split(' ')
        .zip(0..)
        .collect();

    let v_inv: HashMap<usize, &str> = v.iter().map(|(a, b)| (*b, *a)).collect();

    let w = v.len();
    let d = docset.len();
    let k = 2;

    let mut olda = lda::OnlineLDABuilder::new(w, d, k).build();

    let mut perplexity = 0.0;
    for _it in 0..20 {
        for text in docset.iter() {
            let doc = parse_doc(text, &v);
            perplexity = olda.update_lambda_docs(&[doc]);
        }
    }

    assert_eq!(perplexity, 49.734325);

    assert_eq!(vec![
        ("mother", 0.06811236),
        ("brocolli", 0.067197174),
        ("good", 0.06263606),
        ("brother", 0.06259648),
        ("eat", 0.061192695)],
        parse_topic(&olda.get_topic_top_n(0, 5), &v_inv));

    assert_eq!(vec![
        ("health", 0.077420026),
        ("driving", 0.058776595),
        ("experts", 0.04430489),
        ("say", 0.0424421),
        ("tension", 0.0420059)],
        parse_topic(&olda.get_topic_top_n(1, 5), &v_inv));
}
