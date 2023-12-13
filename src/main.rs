#![allow(dead_code)]

mod tokenizer;

use itertools::Itertools;
use regex::Regex;
use std::collections::HashMap;
use std::ops::RangeInclusive;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use tract_onnx::prelude::*;

const PASSAGE: &str = "Google LLC is an American \
multinational technology company that specializes \
in Internet-related services and products, which include \
online advertising technologies, search engine, cloud \
computing, software, and hardware. It is considered one \
of the Big Four technology companies, alongside Amazon, \
Apple, and Facebook. Google was founded in September 1998 \
by Larry Page and Sergey Brin while they were Ph.D. \
students at Stanford University in California. Together \
they own about 14 percent of its shares and control 56 percent \
of the stockholder voting power through supervoting stock. They \
incorporated Google as a California privately held company on \
September 4, 1998, in California. Google was then reincorporated \
in Delaware on October 22, 2002. An initial public offering \
(IPO) took place on August 19, 2004, and Google moved to its \
headquarters in Mountain View, California, nicknamed the \
Googleplex. In August 2015, Google announced plans to reorganize \
its various interests as a conglomerate called Alphabet Inc. \
Google is Alphabet's leading subsidiary and will continue \
to be the umbrella company for Alphabet's Internet interests. \
Sundar Pichai was appointed CEO of Google, replacing Larry Page \
who became the CEO of Alphabet.";

const QUESTION: &str = "Who is the CEO of Google?";

const MAX_TOKENS: usize = 384;
const MAX_QUERY_LEN: usize = 64;
const MAX_ANSWER_LEN: usize = 32;
const PREDICT_ANSWER_NUM: usize = 5;

fn main() -> Result<()> {
    let vocab_path: PathBuf = PathBuf::from("./vocab.txt");
    let vocab = tokenizer::Vocab::create_from_file(&vocab_path)?;
    let bert_tokenizer = tokenizer::BertTokenizer::new(vocab);
    let model = tract_onnx::onnx()
        .model_for_path("./mobilebert.onnx")?
        .into_optimized()?
        .into_runnable()?;

    let start = Instant::now();

    let mut query_tokens = bert_tokenizer.tokenize(QUESTION);
    query_tokens.truncate(MAX_QUERY_LEN);

    let content_words = tokenizer::split_by_whitespace(PASSAGE);
    let mut content_token_idx_to_word_idx_mapping: Vec<usize> = Vec::new();
    let mut content_tokens: Vec<String> = Vec::new();
    for (i, token) in content_words.iter().enumerate() {
        for sub_token in bert_tokenizer.tokenize(&token) {
            content_token_idx_to_word_idx_mapping.push(i);
            content_tokens.push(sub_token);
        }
    }

    // -3 accounts for [CLS], [SEP] and [SEP].
    let max_content_len = MAX_TOKENS - query_tokens.len() - 3;
    content_tokens.truncate(max_content_len);

    let mut tokens: Vec<String> = Vec::new();
    let mut segment_ids: Vec<i32> = Vec::new();

    // Map token index to original index (in feature.origTokens).
    let mut token_idx_to_word_idx_mapping: HashMap<usize, usize> = HashMap::new();

    // Start of generating the `InputFeatures`.
    tokens.push("[CLS]".to_string());
    segment_ids.push(0);

    // For query input.
    for qt in query_tokens {
        tokens.push(qt);
        segment_ids.push(0);
    }

    // For separation.
    tokens.push("[SEP]".to_string());
    segment_ids.push(0);

    // For text input.
    for (i, doc_token) in content_tokens.into_iter().enumerate() {
        tokens.push(doc_token);
        segment_ids.push(1);
        token_idx_to_word_idx_mapping
            .insert(tokens.len(), content_token_idx_to_word_idx_mapping[i]);
    }

    // For ending mark.
    tokens.push("[SEP]".to_string());
    segment_ids.push(1);

    let input_ids = bert_tokenizer.convert_to_ids(&tokens);
    let input_mask = vec![1, input_ids.len()];

    let input_ids_tensor: Tensor = create_input_tensor(
        input_ids
            .iter()
            .map(|i| i32::try_from(*i))
            .collect::<Result<Vec<_>, _>>()?,
    )?;
    let input_mask_tensor: Tensor =
        create_input_tensor(input_mask.into_iter().map(|x| x as i32).collect())?;
    let segment_ids_tensor: Tensor =
        create_input_tensor(segment_ids.into_iter().map(|x| x as i32).collect())?;

    let result = model.run(tvec![
        input_ids_tensor.into(),
        input_mask_tensor.into(),
        segment_ids_tensor.into(),
    ])?;
    let end_logits_tensor: &[f32] = result[0].as_slice()?;
    let start_logits_tensor: &[f32] = result[1].as_slice()?;
    let start_indexes = candidate_answer_indexes(&start_logits_tensor);
    let end_indexes = candidate_answer_indexes(&end_logits_tensor);

    let mut candidates: Vec<_> = start_indexes
        .iter()
        .flat_map(|&start| {
            end_indexes
                .iter()
                .filter_map(|&end| {
                    if start > end {
                        return None;
                    }
                    let logit = start_logits_tensor[start] + end_logits_tensor[end];
                    if end - start > MAX_ANSWER_LEN {
                        return None;
                    }
                    let start_index = token_idx_to_word_idx_mapping.get(&(start + 1));
                    if start_index.is_none() {
                        return None;
                    }
                    let end_index = token_idx_to_word_idx_mapping.get(&(end + 1));
                    if end_index.is_none() {
                        return None;
                    }
                    let &start_index = start_index.unwrap();
                    let &end_index = end_index.unwrap();
                    if start_index > end_index {
                        return None;
                    }
                    Some(Prediction {
                        logit,
                        word_range: start_index..=end_index,
                    })
                })
                .collect_vec()
        })
        .sorted_by(|a, b| b.logit.total_cmp(&a.logit))
        .collect();
    candidates.truncate(PREDICT_ANSWER_NUM);

    let answers = softmaxed(candidates)
        .into_iter()
        .map(|score| {
            let pattern = content_words[score.word_range.clone()]
                .iter()
                .map(|w| regex::escape(w))
                .join("\\s+");
            let re = Regex::new(&pattern)?;
            let m = re
                .find(PASSAGE)
                .context("unable to find pattern in PASSAGE")?;
            Ok(QaAnswer {
                text: m.as_str().to_string(),
                score: score.score,
                logit: score.logit,
                range: score.word_range,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    for a in answers {
        println!("{:?}", a);
    }
    println!("done: {:?}", Instant::now() - start);
    Ok(())
}

fn create_input_tensor(mut v: Vec<i32>) -> Result<Tensor> {
    v.resize(MAX_TOKENS, 0);
    Ok(tract_ndarray::Array2::from_shape_vec((1, MAX_TOKENS), v)?.into())
}

fn candidate_answer_indexes(v: &[f32]) -> Vec<usize> {
    v.iter()
        .take(MAX_TOKENS)
        .enumerate()
        .sorted_by(|(_, a), (_, b)| b.total_cmp(a))
        .take(PREDICT_ANSWER_NUM)
        .map(|(idx, _)| idx)
        .collect()
}

/// Compute softmax probability score over raw logits.
///
/// - Parameter predictions: Array of logit and it range sorted by the logit value in decreasing
///   order.
fn softmaxed(predications: Vec<Prediction>) -> Vec<Score> {
    let max_logit = match predications.first() {
        Some(p) => p.logit,
        None => return Vec::new(),
    };
    let numerators: Vec<_> = predications
        .into_iter()
        .map(|p| {
            let numerator = (p.logit - max_logit).exp();
            (numerator, p)
        })
        .collect();
    let sum: f32 = numerators.iter().map(|(n, _)| n).sum();
    return numerators
        .into_iter()
        .map(|(numerator, prediction)| Score {
            score: numerator / sum,
            logit: prediction.logit,
            word_range: prediction.word_range,
        })
        .collect();
}

struct Prediction {
    pub logit: f32,
    pub word_range: RangeInclusive<usize>,
}

struct Score {
    pub score: f32,
    pub logit: f32,
    pub word_range: RangeInclusive<usize>,
}

#[derive(Debug)]
struct QaAnswer {
    pub text: String,
    pub score: f32,
    pub logit: f32,
    pub range: RangeInclusive<usize>,
}
