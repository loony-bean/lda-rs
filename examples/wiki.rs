extern crate wikipedia;
extern crate regex;
extern crate lda;

use regex::Regex;
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

fn unwiki(content: &str) -> String {
    fn sub(pattern: &str, replacement: &str, text: &str) -> String {
        Regex::new(pattern).unwrap().replace_all(text, replacement).into_owned()
    }

    let mut all = content.to_owned();

    all = sub(r"\n", " ", &all);
    all = sub(r"\{\{.*?\}\}", "", &all);
    all = sub(r"\[\[Category:.*", "", &all);
    all = sub(r"==\s*[Ss]ource\s*==.*", "", &all);
    all = sub(r"==\s*[Rr]eferences\s*==.*", "", &all);
    all = sub(r"==\s*[Ee]xternal [Ll]inks\s*==.*", "", &all);
    all = sub(r"==\s*[Ee]xternal [Ll]inks and [Rr]eferences==\s*", "", &all);
    all = sub(r"==\s*[Ss]ee [Aa]lso\s*==.*", "", &all);
    all = sub(r"http://[^\s]*", "", &all);
    all = sub(r"\[\[Image:.*?\]\]", "", &all);
    all = sub(r"Image:.*?\|", "", &all);
    all = sub(r"\[\[.*?\|*([^\|]*?)\]\]", r"\1", &all);
    all = sub(r"\&lt;.*?&gt;", "", &all);

    all
}

fn get_random_wikipedia_articles(count: usize) -> Vec<(String, String)> {
    let wiki = wikipedia::Wikipedia::<wikipedia::http::default::Client>::default();
    let mut result = Vec::new();
    loop {
        if result.len() >= count {
            break
        }

        for title in &wiki.random_count(10).ok().unwrap_or_default() {
            println!("{}", title);
            let page = wiki.page_from_title(title.clone());
            if let Ok(content) = page.get_content() {
                result.push((title.clone(), content));
            }
        }
    }
    result
}

fn main() {
    // vocabulary
    let text = include_str!("dictnostops.txt");
    let vocab: HashMap<&str, usize> = text.split('\n').zip(0..).collect();
    let vocab2: HashMap<usize, &str> = text.split('\n').enumerate().collect();

    // settings
    let w = vocab.len();
    let d = 3300000;
    let k = 10; // The number of topics

    // init
    let mut olda = lda::OnlineLDABuilder::new(w, d, k).build();

    // feed data
    for it in 1..10 {
        let batch_size = 10;
        let grabs: Vec<_> = get_random_wikipedia_articles(batch_size)
            .into_iter()
            .map(|(_title, text)| unwiki(text.as_str()))
            .collect();

        let docs: Vec<_> = grabs
            .into_iter()
            .map(|text| parse_doc(text.as_str(), &vocab))
            .collect();

        let perplexity = olda.update_lambda_docs(&docs[..]);
        println!("{}: held-out perplexity estimate = {}", it * batch_size, perplexity);
    }

    // print topics
    for idx in 0..k {
        let topic = Topic(olda.get_topic_top_n(idx, 10));
        let topic = topic.translate(&vocab2);
        println!("topic {}:\n{}", idx, topic);
    }
}
