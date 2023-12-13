use anyhow::Result;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use unicode_categories::UnicodeCategories;
use unicode_normalization::UnicodeNormalization;

#[derive(PartialEq, Eq, Clone, Copy)]
enum CaseSensitive {
    Yes,
    No,
}

fn basic_tokenize(text: &str, cs: CaseSensitive) -> Vec<String> {
    let mut cleaned_text = cleaned(&text);
    if cs == CaseSensitive::No {
        cleaned_text = cleaned_text.to_lowercase();
    }
    cleaned_text
        .split_whitespace()
        .flat_map(|s| tokenized_with_punctuation(s))
        .collect()
}

const UNKNOWN_TOKEN: &str = "[UNK]";
const MAX_INPUT_CHARS_PER_WORD: usize = 200;

pub(crate) struct Vocab {
    ids: HashMap<String, usize>,
}

impl Vocab {
    pub fn create_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        Self::create_from_lines(BufReader::new(file).lines())
    }

    fn create_from_lines<B: BufRead>(lines: std::io::Lines<B>) -> Result<Self> {
        Ok(Self {
            ids: lines
                .enumerate()
                .map(|(idx, line)| Ok((line?, idx)))
                .collect::<Result<HashMap<_, _>>>()?,
        })
    }
}

fn wordpiece_tokenizer(vocab: &Vocab, text: &str) -> Vec<String> {
    let mut output_tokens = Vec::new();
    for token in split_by_whitespace(text) {
        if token.len() > MAX_INPUT_CHARS_PER_WORD {
            output_tokens.push(UNKNOWN_TOKEN.to_string());
            continue;
        }
        let mut start = 0;
        let mut sub_words: Vec<String> = Vec::new();
        while start < token.len() {
            let mut end = token.len();
            let mut has_found = false;
            while start < end {
                let mut sub_str: String = token[start..end].to_string();
                if start > 0 {
                    sub_str.insert_str(0, "##");
                }
                if vocab.ids.contains_key(&sub_str) {
                    has_found = true;
                    sub_words.push(sub_str);
                    break;
                } else {
                    end -= 1;
                }
            }
            if has_found {
                start = end;
            } else {
                sub_words = vec![UNKNOWN_TOKEN.to_string()];
                break;
            }
        }
        output_tokens.append(&mut sub_words);
    }
    output_tokens
}

pub(crate) struct BertTokenizer {
    vocab: Vocab,
    cs: CaseSensitive,
}

impl BertTokenizer {
    pub fn new(vocab: Vocab) -> Self {
        return Self {
            vocab,
            cs: CaseSensitive::No,
        };
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        basic_tokenize(text, self.cs)
            .into_iter()
            .flat_map(|t| wordpiece_tokenizer(&self.vocab, &t))
            .collect()
    }

    pub fn convert_to_ids(&self, tokens: &[String]) -> Vec<usize> {
        tokens.iter().map(|t| *self.vocab.ids.get(t).unwrap()).collect()
    }
}

fn is_whitespace_for_bert(c: char) -> bool {
    match c {
        ' ' | '\t' | '\n' | '\r' => true,
        _ => c.is_whitespace(),
    }
}

fn is_control_for_bert(c: char) -> bool {
    if is_whitespace_for_bert(c) {
        return false;
    }
    c.is_other_control() || c.is_other_format()
}

fn should_be_removed_for_bert(c: char) -> bool {
    c == '\0' || c == '\u{fffd}'
}

fn is_punctuation_for_bert(c: char) -> bool {
    if c.is_ascii() && c as u8 > 32 && !c.is_alphabetic() && !c.is_numeric() {
        return true;
    }
    c.is_punctuation_close()
        || c.is_punctuation_connector()
        || c.is_punctuation_dash()
        || c.is_punctuation_final_quote()
        || c.is_punctuation_initial_quote()
        || c.is_punctuation_open()
        || c.is_punctuation_other()
}

/// Performs invalid character removal and whitespace cleanup on text.
///
/// Replaces all whitespace code points with spaces and control characters including \t, \n, \r.
///
/// - Returns: Cleaned text.
fn cleaned(text: &str) -> String {
    text.nfc()
        .filter_map(|scalar| {
            if is_whitespace_for_bert(scalar) {
                Some(' ')
            } else if is_control_for_bert(scalar) {
                None
            } else if should_be_removed_for_bert(scalar) {
                None
            } else {
                Some(scalar)
            }
        })
        .collect::<String>()
}

pub(crate) fn split_by_whitespace(text: &str) -> Vec<String> {
    text.nfc()
        .fold(vec![], |mut acc, c| {
            if is_whitespace_for_bert(c) {
                acc.push(String::new());
            } else {
                match acc.last_mut() {
                    Some(l) => l.push(c),
                    None => return vec![c.to_string()],
                }
            }
            acc
        })
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect()
}

fn tokenized_with_punctuation(text: &str) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    let mut current_token: String = String::new();
    for unicode in text.nfc() {
        if is_punctuation_for_bert(unicode) {
            if !current_token.is_empty() {
                tokens.push(std::mem::take(&mut current_token));
            }
            tokens.push(unicode.to_string());
        } else {
            current_token.push(unicode);
        }
    }
    if !current_token.is_empty() {
        tokens.push(current_token);
    }
    tokens
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{Vocab, wordpiece_tokenizer};

    #[test]
    fn wordpiece_tokenizer_test() {
        let vocab = Vocab::create_from_file(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("vocab.txt")).unwrap();

        let input = "unaffable";
        let output = vec!["una", "##ffa", "##ble"];
        assert_eq!(wordpiece_tokenizer(&vocab, input), output);

        let input = "unaffableX";
        let output = vec!["[UNK]"];
        assert_eq!(wordpiece_tokenizer(&vocab, input), output);
    }
}
