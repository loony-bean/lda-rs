extern crate wikipedia;

extern crate itertools;
extern crate regex;
extern crate ordermap;
extern crate ndarray;

extern crate lda;

use std::path::{Path, PathBuf};
use std::fs;
use std::fs::File;
use std::io;
use std::io::Read;
use std::collections::hash_map::HashMap;
use std::fmt::{Display, Formatter, Error as FmtError};
use std::hash::Hash;

struct Topic<T> (Vec<(T, f32)>);

impl<T> Topic<T> where T: Eq + Hash {
    fn translate<'a, A>(&self, vocab: &'a HashMap<T, A>) -> Topic<&'a A> {
        Topic(self.0.iter()
            .map(|&(ref idx, p)| (&vocab[idx], p))
            .collect())
    }
}

impl<T: Display> Display for Topic<T> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), FmtError> {
        for &(ref k, p) in self.0.iter() {
            write!(f, "  {0: <20}  \t---\t  {1:.4}\n", k, p)?;
        }
        Ok(())
    }
}

fn parse_doc(text: &str, vocab: &HashMap<&str, usize>) -> lda::Document {
    text
        .split(|c: char| !c.is_alphabetic())
        .map(|s| s.to_lowercase())
        .filter_map(|s| vocab.get(&*s))
        .collect()
}

fn read_file(path: &Path) -> Result<String, io::Error> {
    let mut f = File::open(path)?;
    let mut text = String::new();
    f.read_to_string(&mut text)?;
    Ok(text)
}

fn read_dir(path: &str) -> Result<Vec<PathBuf>, io::Error> {
    let dir = fs::read_dir(path)?;
    let mut result = Vec::new();
    for entry in dir {
        result.push(entry?.path());
    }
    Ok(result)
}

fn main() {
    // vocabulary
    let text = include_str!("dictnostops.txt");

    let vocab: HashMap<&str, usize> = text.split('\n').zip((0..)).collect();
    let vocab2: HashMap<usize, &str> = text.split('\n').enumerate().collect();

    // settings
    let w = vocab.len();
    let d = 1000;
    let k = 10; // The number of topics

    // init
    let mut olda = lda::OnlineLDABuilder::new(w, d, k).build();

    // feed data
    let dir = read_dir("./examples/data").expect("found example data");
    for (it, path) in dir.iter().take(d).enumerate() {
        let text = read_file(&path).expect("read file content");
        let doc = parse_doc(text.as_str(), &vocab);

        let perplexity = olda.update_lambda_docs(&[doc]);
        println!("{}: held-out perplexity estimate = {}", it, perplexity);
    }

    // print topics
    for idx in 0..k {
        let topic = Topic(olda.get_topic_top_n(idx, 10));
        let topic = topic.translate(&vocab2);
        println!("topic {}:\n{}", idx, topic);
    }
}
