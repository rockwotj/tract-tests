#![allow(dead_code)]

mod tokenizer;

use itertools::Itertools;
use regex::Regex;
use std::ops::RangeInclusive;
use std::time::Instant;
use std::{collections::HashMap, path::Path};

use anyhow::{bail, Context, Result};
use tract_onnx::prelude::*;

const MAX_SEQ_LEN: usize = 384;
const MAX_QUERY_LEN: usize = 64;
const MAX_ANS_LEN: usize = 32;
const PREDICT_ANS_NUM: usize = 5;
const OUTPUT_OFFSET: usize = 1;

struct BertQuestionAnswerer {
    tokenizer: tokenizer::BertTokenizer,
    model: TypedRunnableModel<Graph<TypedFact, Box<dyn TypedOp>>>,
}

impl BertQuestionAnswerer {
    pub fn new_from_files(vocab: impl AsRef<Path>, model: impl AsRef<Path>) -> Result<Self> {
        let vocab = tokenizer::Vocab::create_from_file(&vocab)?;
        let tokenizer = tokenizer::BertTokenizer::new(vocab);
        let model = match model.as_ref().extension().and_then(|s| s.to_str()) {
            Some("tflite") => tract_tflite::tflite()
                .model_for_path(model)?
                .into_optimized()?
                .into_runnable()?,
            Some("onnx") => tract_onnx::onnx()
                .model_for_path(model)?
                .into_optimized()?
                .into_runnable()?,
            Some("nnef") => tract_nnef::nnef()
                .model_for_path(model)?
                .into_optimized()?
                .into_runnable()?,
            _ => bail!("unexpected model file extension {:?}", model.as_ref().extension()),
        };
        return Ok(Self { tokenizer, model });
    }

    pub fn answer(&self, content: &str, question: &str) -> Result<Vec<QaAnswer>> {
        let (input_ids, input_mask, segment_ids, content_data) =
            self.preprocess(content, question)?;
        println!("INPUT_IDS:\n{}", input_ids.dump(true)?);
        println!("\nINPUT_MASK:\n{}", input_mask.dump(true)?);
        println!("\nSEGMENT_IDS:\n{}\n\n", segment_ids.dump(true)?);
        let result = self.model.run(tvec![
            input_ids.into(),
            input_mask.into(),
            segment_ids.into(),
        ])?;
        println!("END_LOGITS:\n{}", result[0].dump(true)?);
        println!("\nSTART_LOGITS:\n{}\n\n", result[1].dump(true)?);
        let end_logits_tensor: &[f32] = result[0].as_slice()?;
        let start_logits_tensor: &[f32] = result[1].as_slice()?;
        self.postprocess(content_data, start_logits_tensor, end_logits_tensor)
    }

    fn preprocess<'a>(
        &self,
        content: &'a str,
        question: &str,
    ) -> Result<(Tensor, Tensor, Tensor, ContentData<'a>)> {
        let mut query_tokens = self.tokenizer.tokenize(question);
        query_tokens.truncate(MAX_QUERY_LEN);

        let content_words = tokenizer::split_by_whitespace(content);
        let mut content_token_idx_to_word_idx_mapping: Vec<usize> = Vec::new();
        let mut content_tokens: Vec<String> = Vec::new();
        for (i, token) in content_words.iter().enumerate() {
            for sub_token in self.tokenizer.tokenize(&token) {
                content_token_idx_to_word_idx_mapping.push(i);
                content_tokens.push(sub_token);
            }
        }

        // -3 accounts for [CLS], [SEP] and [SEP].
        let max_content_len = MAX_SEQ_LEN - query_tokens.len() - 3;
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

        let input_ids = self.tokenizer.convert_to_ids(&tokens);
        let input_mask = vec![1; input_ids.len()];

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
        Ok((
            input_ids_tensor,
            input_mask_tensor,
            segment_ids_tensor,
            ContentData {
                content_words,
                token_idx_to_word_idx_mapping,
                original_content: content,
            },
        ))
    }

    fn postprocess(
        &self,
        content_data: ContentData,
        start_logits: &[f32],
        end_logits: &[f32],
    ) -> Result<Vec<QaAnswer>> {
        let start_indexes = candidate_answer_indexes(&start_logits);
        let end_indexes = candidate_answer_indexes(&end_logits);
        let mut candidates: Vec<_> = start_indexes
            .iter()
            .flat_map(|&start| {
                end_indexes
                    .iter()
                    .filter_map(|&end| {
                        let logit = start_logits[start] + end_logits[end];
                        self.create_prediction(
                            logit,
                            start,
                            end,
                            &content_data.token_idx_to_word_idx_mapping,
                        )
                    })
                    .collect_vec()
            })
            .sorted_by(|a, b| b.logit.total_cmp(&a.logit))
            .collect();
        candidates.truncate(PREDICT_ANS_NUM);

        softmaxed(candidates)
            .into_iter()
            .map(|score| {
                let pattern = content_data.content_words[score.word_range.clone()]
                    .iter()
                    .map(|w| regex::escape(w))
                    .join("\\s+");
                let re = Regex::new(&pattern)?;
                let m = re
                    .find(content_data.original_content)
                    .context("unable to find pattern in original content")?;
                Ok(QaAnswer {
                    text: m.as_str().to_string(),
                    score: score.score,
                    logit: score.logit,
                    range: score.word_range,
                })
            })
            .collect::<Result<Vec<_>>>()
    }

    fn create_prediction(
        &self,
        logit: f32,
        start: usize,
        end: usize,
        mapping: &HashMap<usize, usize>,
    ) -> Option<Prediction> {
        if end < start {
            return None;
        }
        if (end - start + 1) > MAX_ANS_LEN {
            return None;
        }
        let start_index = mapping.get(&(start + OUTPUT_OFFSET));
        if start_index.is_none() {
            return None;
        }
        let end_index = mapping.get(&(end + OUTPUT_OFFSET));
        if end_index.is_none() {
            return None;
        }
        let &start_index = start_index.unwrap();
        let &end_index = end_index.unwrap();
        if end_index < start_index {
            return None;
        }
        Some(Prediction {
            logit,
            word_range: start_index..=end_index,
        })
    }
}

fn main() -> Result<()> {
    let oracle = BertQuestionAnswerer::new_from_files("./vocab.txt", "./mobilebert-opt.nnef")?;
    const PASSAGE: &str = "TensorFlow is a free and open-source software library for dataflow and \
differentiable programming across a range of tasks. It is a symbolic math library, and \
is also used for machine learning applications such as neural networks. It is used for \
both research and production at Google. TensorFlow was developed by the Google Brain \
team for internal Google use. It was released under the Apache License 2.0 on November \
9, 2015.";
    const QUESTION: &str = "Who developed TensorFlow?";
    let start = Instant::now();
    for a in oracle.answer(PASSAGE, QUESTION)? {
        println!("{:?}", a);
    }
    println!("done: {:?}", Instant::now() - start);
    Ok(())
}

fn create_input_tensor(mut v: Vec<i32>) -> Result<Tensor> {
    v.resize(MAX_SEQ_LEN, 0);
    Ok(tract_ndarray::Array2::from_shape_vec((1, MAX_SEQ_LEN), v)?.into())
}

fn candidate_answer_indexes(v: &[f32]) -> Vec<usize> {
    v.iter()
        .take(MAX_SEQ_LEN)
        .enumerate()
        .sorted_by(|(_, a), (_, b)| b.total_cmp(a))
        .take(PREDICT_ANS_NUM)
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

struct ContentData<'a> {
    pub content_words: Vec<String>,
    pub token_idx_to_word_idx_mapping: HashMap<usize, usize>,
    pub original_content: &'a str,
}

#[derive(Debug)]
struct QaAnswer {
    pub text: String,
    pub score: f32,
    pub logit: f32,
    pub range: RangeInclusive<usize>,
}
